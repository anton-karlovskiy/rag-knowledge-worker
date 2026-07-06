import gradio as gr
from dotenv import load_dotenv

from answer import answer_question

load_dotenv(override=True)


def format_context(context_chunks):
    formatted = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for chunk in context_chunks:
        formatted += f"<span style='color: #ff7800;'>Source: {chunk.metadata['source']}</span>\n\n"
        formatted += chunk.page_content + "\n\n"
    return formatted


def chat(history):
    latest_question = history[-1]["content"]
    prior_messages = history[:-1]
    answer, context_chunks = answer_question(latest_question, prior_messages)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context_chunks)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant", theme=theme) as ui:
        gr.Markdown("# Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation", height=600, type="messages", show_copy_button=True
                )
                message = gr.Textbox(
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

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
