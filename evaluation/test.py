import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_CASES_FILE = str(Path(__file__).parent / "test_cases.jsonl")


class TestCase(BaseModel):
    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


def load_test_cases() -> list[TestCase]:
    test_cases = []
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            data = json.loads(line.strip())
            test_cases.append(TestCase(**data))
    return test_cases
