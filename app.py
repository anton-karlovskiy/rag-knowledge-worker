from pathlib import PurePath

import gradio as gr
from dotenv import load_dotenv

from pipelines import DEFAULT_PIPELINE, PIPELINE_LABELS, get_pipeline

load_dotenv(override=True)


def display_source(source: str) -> str:
    """
    Show the source path relative to knowledge-base/, hiding the local directory it was ingested from.
    """
    parts = PurePath(source).parts
    if "knowledge-base" not in parts:
        return source
    return "/".join(parts[parts.index("knowledge-base") + 1:])


def format_context(context_chunks):
    formatted = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for chunk in context_chunks:
        formatted += f"<span style='color: #ff7800;'>Source: {display_source(chunk.metadata['source'])}</span>\n\n"
        formatted += chunk.page_content + "\n\n"
    return formatted


def content_to_text(content) -> str:
    """
    Gradio 6 stores message content as a list of blocks, e.g. [{"type": "text", "text": "..."}].
    """
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content if block.get("type") == "text")


def chat(history, pipeline_name):
    messages = [{"role": message["role"], "content": content_to_text(message["content"])} for message in history]
    latest_question = messages[-1]["content"]
    prior_messages = messages[:-1]
    answer, context_chunks = get_pipeline(pipeline_name).answer_question(latest_question, prior_messages)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context_chunks)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# Insurellm Expert Assistant\nAsk me anything about Insurellm!")
        pipeline_selector = gr.Radio(
            choices=[(label, name) for name, label in PIPELINE_LABELS.items()],
            value=DEFAULT_PIPELINE,
            label="RAG Pipeline",
        )

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation", height=600, buttons=["copy"]
                )
                question_box = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        question_box.submit(
            put_message_in_chatbot, inputs=[question_box, chatbot], outputs=[question_box, chatbot]
        ).then(chat, inputs=[chatbot, pipeline_selector], outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()
