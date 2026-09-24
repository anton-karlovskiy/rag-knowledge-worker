import sys
import math
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv

from evaluation.test import TestCase, load_test_cases
from answer import answer_question, fetch_context
from config import RETRIEVAL_K


load_dotenv(override=True)

MODEL = "openai/gpt-4.1-nano"


class RetrievalEval(BaseModel):
    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    mean_ndcg: float = Field(
        description="Normalized Discounted Cumulative Gain (binary relevance) - average across all keywords"
    )
    found_keywords: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage_percent: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong) to 5 (ideal)."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor) to 5 (ideal)."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor) to 5 (ideal)."
    )


# RR = Reciprocal Rank: 1/rank of the first retrieved doc containing the keyword
def calculate_rr(keyword: str, retrieved_docs: list) -> float:
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int) -> float:
    keyword_lower = keyword.lower()
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]
    ]
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test_case: TestCase) -> RetrievalEval:
    retrieved_docs = fetch_context(test_case.question)
    reciprocal_ranks = [calculate_rr(keyword, retrieved_docs) for keyword in test_case.keywords]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, RETRIEVAL_K) for keyword in test_case.keywords]
    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    found_keywords = sum(1 for reciprocal_rank in reciprocal_ranks if reciprocal_rank > 0)
    total_keywords = len(test_case.keywords)
    keyword_coverage_percent = (found_keywords / total_keywords * 100) if total_keywords > 0 else 0.0
    return RetrievalEval(
        mrr=mrr,
        mean_ndcg=mean_ndcg,
        found_keywords=found_keywords,
        total_keywords=total_keywords,
        keyword_coverage_percent=keyword_coverage_percent,
    )


def evaluate_answer(test_case: TestCase) -> tuple[AnswerEval, str, list]:
    generated_answer, retrieved_docs = answer_question(test_case.question)
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test_case.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test_case.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]
    judge_response = completion(model=MODEL, messages=judge_messages, response_format=AnswerEval)
    answer_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content)
    return answer_eval, generated_answer, retrieved_docs


def evaluate_retrieval_all():
    test_cases = load_test_cases()
    for index, test_case in enumerate(test_cases):
        retrieval_eval = evaluate_retrieval(test_case)
        yield test_case, retrieval_eval, (index + 1) / len(test_cases)


def evaluate_answer_all():
    test_cases = load_test_cases()
    for index, test_case in enumerate(test_cases):
        answer_eval = evaluate_answer(test_case)[0]
        yield test_case, answer_eval, (index + 1) / len(test_cases)


def run_cli_evaluation(test_case_index: int):
    test_cases = load_test_cases()

    if test_case_index < 0 or test_case_index >= len(test_cases):
        print(f"Error: test_case_index must be between 0 and {len(test_cases) - 1}")
        sys.exit(1)

    test_case = test_cases[test_case_index]

    print(f"\n{'=' * 80}")
    print(f"Test Case #{test_case_index}")
    print(f"{'=' * 80}")
    print(f"Question: {test_case.question}")
    print(f"Keywords: {test_case.keywords}")
    print(f"Category: {test_case.category}")
    print(f"Reference Answer: {test_case.reference_answer}")

    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")
    retrieval_eval = evaluate_retrieval(test_case)
    print(f"MRR: {retrieval_eval.mrr:.4f}")
    print(f"Mean nDCG: {retrieval_eval.mean_ndcg:.4f}")
    print(f"Keywords Found: {retrieval_eval.found_keywords}/{retrieval_eval.total_keywords}")
    print(f"Keyword Coverage: {retrieval_eval.keyword_coverage_percent:.1f}%")

    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")
    answer_eval, generated_answer, _ = evaluate_answer(test_case)
    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_eval.feedback}")
    print("\nScores:")
    print(f"  Accuracy:     {answer_eval.accuracy:.2f}/5")
    print(f"  Completeness: {answer_eval.completeness:.2f}/5")
    print(f"  Relevance:    {answer_eval.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run eval <test_case_index>")
        sys.exit(1)
    try:
        test_case_index = int(sys.argv[1])
    except ValueError:
        print("Error: test_case_index must be an integer")
        sys.exit(1)
    run_cli_evaluation(test_case_index)


if __name__ == "__main__":
    main()
