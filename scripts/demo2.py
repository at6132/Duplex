"""
Chat-style demo: talk to Duplex-1.2-1.7B like ChatGPT.
Single conversation panel, no comparison. For normal back-and-forth chat.

Usage:
    python scripts/demo2.py --duplex_ckpt checkpoints/duplex-1.2-1.7b/final.pt
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from duplex.inference.generate import load_duplex_model, load_vanilla_qwen, generate_vanilla


# Conversation format for the model
USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant: "

# Instruct both models to give a single reply and not continue the conversation
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Reply in one direct message. "
    "When you receive new information or a correction mid-response, briefly acknowledge it and then continue with the updated information. "
    "Do not generate any extra 'User:' or 'Assistant:' lines—only your single reply.\n\n"
)


def build_prompt_from_messages(history: list[dict], new_message: str) -> str:
    """Turn message history + new message into a single prompt string."""
    parts = [SYSTEM_INSTRUCTION.strip()]
    for m in history:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"{USER_PREFIX}{content}")
        elif role == "assistant":
            parts.append(f"{ASSISTANT_PREFIX}{content}")
    parts.append(f"{USER_PREFIX}{new_message.strip()}")
    parts.append(ASSISTANT_PREFIX.rstrip())
    return "\n\n".join(parts)


def extract_reply(full_text: str, prompt: str) -> str:
    """Get only the assistant's reply; stop at any new 'User:' line (model hallucination)."""
    if full_text.startswith(prompt):
        reply = full_text[len(prompt) :].strip()
    else:
        idx = full_text.rfind(ASSISTANT_PREFIX)
        reply = full_text[idx + len(ASSISTANT_PREFIX) :].strip() if idx >= 0 else full_text.strip()

    # Keep only the first assistant turn: truncate at next "User:" (runaway generation)
    for stop in ("\n\nUser:", "\nUser:"):
        if stop in reply:
            reply = reply.split(stop)[0]
    reply = reply.strip()
    # Drop leading "Assistant:" if the model echoed it
    if reply.lower().startswith("assistant:"):
        reply = reply[10:].strip()
    return reply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duplex_ckpt", required=True, help="Path to Duplex checkpoint (e.g. checkpoints/duplex-1.2-1.7b/final.pt)")
    parser.add_argument("--qwen_path", default="models/qwen3-1.7b-base")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print("Loading Qwen3-1.7B (vanilla)...")
    qwen_model, qwen_tokenizer = load_vanilla_qwen(args.qwen_path, quantize=False)
    print("Loading Duplex-1.2-1.7B...")
    duplex_model = load_duplex_model(args.qwen_path, args.duplex_ckpt)
    print("Ready.\n")

    MODEL_CHOICES = ["Duplex-1.2-1.7B", "Qwen3-1.7B (Vanilla)"]

    def chat(message: str, history: list, model_choice: str):
        if not message.strip():
            return history or []

        history = list(history or [])
        prompt = build_prompt_from_messages(history, message)

        if model_choice == "Qwen3-1.7B (Vanilla)":
            full_text = generate_vanilla(
                qwen_model, qwen_tokenizer,
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        else:
            full_text, _ = duplex_model.generate_with_update(
                prompt_text=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )

        reply = extract_reply(full_text, prompt)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return history

    with gr.Blocks(title="Duplex-1 Chat") as demo:
        gr.Markdown("# Chat\nChoose a model and ask anything.")
        model_dropdown = gr.Dropdown(
            choices=MODEL_CHOICES,
            value=MODEL_CHOICES[0],
            label="Model",
        )
        chatbot = gr.Chatbot(
            label="Chat",
            height=500,
        )
        msg = gr.Textbox(
            placeholder="Type a message...",
            label="Message",
            lines=2,
            scale=7,
            show_label=False,
            container=False,
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

        def respond(message, history, model_choice):
            if not message or not message.strip():
                return history or []
            return chat(message, history or [], model_choice)

        msg.submit(respond, [msg, chatbot, model_dropdown], [chatbot])
        submit_btn.click(respond, [msg, chatbot, model_dropdown], [chatbot])
        msg.submit(lambda: "", None, [msg])

    print(f"Chat demo: http://localhost:{args.port}")
    demo.launch(
        server_port=args.port,
        share=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
