import gradio as gr
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

from evaluation.eval import evaluate_retrieval_all, evaluate_answer_all

load_dotenv(override=True)

# Color coding thresholds: (green, amber). Below amber is red.
THRESHOLDS = {
    "mrr": (0.9, 0.75),
    "ndcg": (0.9, 0.75),
    "keyword-coverage": (90.0, 75.0),
    # Answer metrics use a 1-5 scale
    "accuracy": (4.5, 4.0),
    "completeness": (4.5, 4.0),
    "relevance": (4.5, 4.0),
}

PLACEHOLDER_HTML = (
    "<div style='padding: 20px; text-align: center; color: #999;'>"
    "Click 'Run Evaluation' to start</div>"
)


def get_color(value: float, metric_type: str) -> str:
    if metric_type not in THRESHOLDS:
        return "black"
    green, amber = THRESHOLDS[metric_type]
    if value >= green:
        return "green"
    if value >= amber:
        return "orange"
    return "red"


def format_metric_html(
    label: str,
    value: float,
    metric_type: str,
    is_percentage: bool = False,
    score_format: bool = False,
) -> str:
    color = get_color(value, metric_type)
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>
    """


def format_complete_html(count: int) -> str:
    return f"""
    <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
        <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} test cases</span>
    </div>
    """


def run_retrieval_evaluation(progress=gr.Progress()):
    total_mrr = 0.0
    total_mean_ndcg = 0.0
    total_keyword_coverage_percent = 0.0
    category_mrr = defaultdict(list)
    count = 0

    for test_case, retrieval_eval, progress_value in evaluate_retrieval_all():
        count += 1
        total_mrr += retrieval_eval.mrr
        total_mean_ndcg += retrieval_eval.mean_ndcg
        total_keyword_coverage_percent += retrieval_eval.keyword_coverage_percent
        category_mrr[test_case.category].append(retrieval_eval.mrr)
        progress(progress_value, desc=f"Evaluating test case {count}...")

    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", total_mrr / count, "mrr")}
        {format_metric_html("Normalized DCG (nDCG)", total_mean_ndcg / count, "ndcg")}
        {format_metric_html("Keyword Coverage", total_keyword_coverage_percent / count, "keyword-coverage", is_percentage=True)}
        {format_complete_html(count)}
    </div>
    """

    category_df = pd.DataFrame(
        [
            {"Category": category, "Average MRR": sum(scores) / len(scores)}
            for category, scores in category_mrr.items()
        ]
    )
    return final_html, category_df


def run_answer_evaluation(progress=gr.Progress()):
    total_accuracy = 0.0
    total_completeness = 0.0
    total_relevance = 0.0
    category_accuracy = defaultdict(list)
    count = 0

    for test_case, answer_eval, progress_value in evaluate_answer_all():
        count += 1
        total_accuracy += answer_eval.accuracy
        total_completeness += answer_eval.completeness
        total_relevance += answer_eval.relevance
        category_accuracy[test_case.category].append(answer_eval.accuracy)
        progress(progress_value, desc=f"Evaluating test case {count}...")

    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Accuracy", total_accuracy / count, "accuracy", score_format=True)}
        {format_metric_html("Completeness", total_completeness / count, "completeness", score_format=True)}
        {format_metric_html("Relevance", total_relevance / count, "relevance", score_format=True)}
        {format_complete_html(count)}
    </div>
    """

    category_df = pd.DataFrame(
        [
            {"Category": category, "Average Accuracy": sum(scores) / len(scores)}
            for category, scores in category_accuracy.items()
        ]
    )
    return final_html, category_df


def main():
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Evaluation Dashboard", theme=theme) as ui:
        gr.Markdown("# RAG Evaluation Dashboard")
        gr.Markdown("Evaluate retrieval and answer quality for the Insurellm RAG system")

        gr.Markdown("## Retrieval Evaluation")
        retrieval_button = gr.Button("Run Evaluation", variant="primary", size="lg")
        with gr.Row():
            with gr.Column(scale=1):
                retrieval_metrics = gr.HTML(PLACEHOLDER_HTML)
            with gr.Column(scale=1):
                retrieval_chart = gr.BarPlot(
                    x="Category",
                    y="Average MRR",
                    title="Average MRR by Category",
                    y_lim=[0, 1],
                    height=400,
                )

        gr.Markdown("## Answer Evaluation")
        answer_button = gr.Button("Run Evaluation", variant="primary", size="lg")
        with gr.Row():
            with gr.Column(scale=1):
                answer_metrics = gr.HTML(PLACEHOLDER_HTML)
            with gr.Column(scale=1):
                answer_chart = gr.BarPlot(
                    x="Category",
                    y="Average Accuracy",
                    title="Average Accuracy by Category",
                    y_lim=[1, 5],
                    height=400,
                )

        retrieval_button.click(
            fn=run_retrieval_evaluation, outputs=[retrieval_metrics, retrieval_chart]
        )
        answer_button.click(fn=run_answer_evaluation, outputs=[answer_metrics, answer_chart])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
