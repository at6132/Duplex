"""
Investor demo: side-by-side streaming comparison of vanilla Qwen vs Duplex-1.2-1.7B.

Usage:
    python scripts/demo.py --duplex_ckpt checkpoints/duplex-1.2-1.7b/final.pt
"""

import argparse
import os
import sys
import threading
import time
from queue import Queue, Empty

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gradio as gr

from duplex.inference.generate import (
    load_vanilla_qwen,
    load_duplex_model,
    generate_vanilla_with_restart_streaming,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly and concisely. "
    "When you receive new information or a correction mid-response, briefly acknowledge it and then continue with the updated information. "
    "Do not generate any extra conversation turns.\n\n"
)


def wrap_prompt(raw_prompt: str) -> str:
    return SYSTEM_PROMPT + raw_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duplex_ckpt", required=True)
    parser.add_argument("--qwen_path", default="models/qwen3-1.7b-base")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Loading vanilla Qwen3-1.7B (baseline)...")
    qwen_model, qwen_tokenizer = load_vanilla_qwen(args.qwen_path, quantize=False)

    print("Loading Duplex-1.2-1.7B...")
    duplex_model = load_duplex_model(args.qwen_path, args.duplex_ckpt)
    print("Models loaded.\n")

    def run_comparison(prompt, correction, tokens_before_correction, max_tokens, temperature):
        tokens_before = int(tokens_before_correction)
        max_tok = int(max_tokens)
        temp = float(temperature)
        full_prompt = wrap_prompt(prompt)

        state = {"baseline": "", "duplex": "", "done_b": False, "done_d": False}
        times = {"baseline": 0.0, "duplex": 0.0}
        queue = Queue()

        def run_baseline():
            t0 = time.perf_counter()
            try:
                for text in generate_vanilla_with_restart_streaming(
                    qwen_model, qwen_tokenizer,
                    full_prompt, correction,
                    tokens_before_correction=tokens_before,
                    max_new_tokens=max_tok,
                    temperature=temp,
                ):
                    state["baseline"] = text
                    queue.put(True)
            finally:
                times["baseline"] = time.perf_counter() - t0
                state["done_b"] = True
                queue.put(True)

        def run_duplex():
            t0 = time.perf_counter()
            try:
                for text in duplex_model.generate_with_update_streaming(
                    prompt_text=full_prompt,
                    max_new_tokens=max_tok,
                    temperature=temp,
                    correction_text=correction,
                    correction_after_tokens=tokens_before,
                ):
                    state["duplex"] = text
                    queue.put(True)
            finally:
                times["duplex"] = time.perf_counter() - t0
                state["done_d"] = True
                queue.put(True)

        threading.Thread(target=run_baseline, daemon=True).start()
        threading.Thread(target=run_duplex, daemon=True).start()

        while not (state["done_b"] and state["done_d"]):
            try:
                queue.get(timeout=0.05)
            except Empty:
                pass
            yield (
                state["baseline"],
                state["duplex"],
                "Generating..." if not state["done_b"] else f"Finished in {times['baseline']:.1f}s",
                "Generating..." if not state["done_d"] else f"Finished in {times['duplex']:.1f}s",
            )

        yield (
            state["baseline"],
            state["duplex"],
            f"Finished in {times['baseline']:.1f}s",
            f"Finished in {times['duplex']:.1f}s",
        )

    examples = [
        # ---- Long-form (~2K token): inject EARLY (60–120 tokens) so injection always fires ----
        [
            "Write a detailed step-by-step tutorial on building a REST API in Python with FastAPI. Cover: project setup, defining routes, request validation with Pydantic, error handling, testing, and deployment. Be thorough so a beginner can follow along.",
            "Actually, switch to Flask instead of FastAPI. Keep the same structure but use Flask and Flask-RESTful or plain route decorators.",
            80, 2048, 0.5,
        ],
        [
            "Write a comprehensive guide to the machine learning pipeline: data loading and cleaning, feature engineering, model selection (e.g. linear models, trees, neural nets), training loops, validation, and deployment. Include brief code snippets where helpful.",
            "Use PyTorch for all code examples instead of TensorFlow. Keep the same pipeline structure.",
            90, 2048, 0.5,
        ],
        [
            "Explain the full process of how a CPU executes instructions: fetch, decode, execute, memory access, writeback. Then describe pipelining and basic hazards. Be detailed enough for a computer architecture course.",
            "Add a section on out-of-order execution and the role of the reorder buffer.",
            100, 2048, 0.5,
        ],
        [
            "Write a step-by-step tutorial on building a React app: components, hooks (useState, useEffect), state management, and fetching data from an API. Include a small example app.",
            "Switch to Vue 3 with the Composition API instead of React. Keep the same app structure and concepts.",
            70, 2048, 0.5,
        ],
        [
            "Give a thorough overview of TCP/IP, HTTP/1.1, and TLS: handshakes, headers, and how they work together. Aim for a reader who knows basic networking.",
            "Add a section on QUIC and HTTP/3 and how they differ from TCP/TLS.",
            85, 2048, 0.5,
        ],
        [
            "List and explain the top 15 software design patterns (e.g. Singleton, Factory, Observer, Strategy) with short code examples and when to use each.",
            "Replace the last 5 patterns with concurrency patterns: mutex, semaphore, condition variable, reader-writer lock, and message queue. Include brief examples.",
            110, 2048, 0.5,
        ],
        [
            "Write a step-by-step guide to setting up CI/CD with GitHub Actions: triggers, jobs, steps, and deploying to a cloud provider. Include a sample workflow file.",
            "Use GitLab CI instead of GitHub Actions. Include Docker in the pipeline and a sample .gitlab-ci.yml.",
            75, 2048, 0.5,
        ],
        [
            "Describe the history of the Roman Empire from its founding through the fall of the West. Cover major periods, emperors, and events in order.",
            "Focus the rest of the answer only on the period from Augustus through the Five Good Emperors (96–180 CE). Go into more detail on that span.",
            95, 2048, 0.5,
        ],
        [
            "Explain how photosynthesis works in plants: light and dark reactions, chloroplasts, and the Calvin cycle. Then briefly compare C3 and C4 plants.",
            "Switch to explaining cellular respiration instead: glycolysis, Krebs cycle, and oxidative phosphorylation.",
            65, 2048, 0.5,
        ],
        # ---- Short examples ----
        ["What is the capital of Australia?",
         "That's incorrect. The capital is Canberra, not Sydney.",
         8, 120, 0.5],
        ["Given x = 5, compute 3*x + 7 step by step.",
         "Actually, x = 12 not 5. Please recalculate.",
         8, 120, 0.5],
        ["Write a profile for Maria, 28, architect in Berlin who enjoys painting.",
         "Update: Maria is 35, a data scientist in London, and is learning Mandarin.",
         8, 150, 0.6],
        ["Translate 'The quick brown fox jumps over the lazy dog' into French.",
         "Translate it into Japanese instead and include romaji.",
         6, 100, 0.5],
        ["Summarize the causes of World War I.",
         "Focus only on the assassination of Archduke Franz Ferdinand.",
         10, 150, 0.6],
    ]

    CSS = """
    .header-row { text-align: center; margin-bottom: 8px; }
    .header-row h1 { font-size: 2rem; margin-bottom: 4px; }
    .header-row p  { color: #888; font-size: 0.95rem; }
    .model-card   { border: 1px solid #333; border-radius: 12px; padding: 16px; }
    .baseline-hdr { color: #f87171; font-size: 1.1rem; font-weight: 600; }
    .duplex-hdr   { color: #4ade80; font-size: 1.1rem; font-weight: 600; }
    .status-text  { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .examples-row { margin-top: 12px; }
    """

    with gr.Blocks(title="Duplex-1 Demo") as demo:

        # Header
        gr.HTML("""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <h1 style="font-size:2.2rem; margin:0;">Duplex-1.2-1.7B</h1>
            <p style="color:#aaa; font-size:1rem; margin:6px 0 0 0;">
                Full-Duplex Language Model &mdash; accepts corrections mid-generation without restarting
            </p>
        </div>
        """)

        # How it works banner
        gr.HTML("""
        <div style="display:flex; justify-content:center; gap:40px; padding:10px 20px 20px 20px; flex-wrap:wrap;">
            <div style="text-align:center; max-width:280px;">
                <div style="font-size:1.6rem;">&#9724;</div>
                <div style="font-weight:600; color:#f87171;">Baseline (Qwen)</div>
                <div style="color:#999; font-size:0.85rem;">Generates &rarr; STOPS &rarr; Restarts from scratch with correction</div>
            </div>
            <div style="text-align:center; max-width:280px;">
                <div style="font-size:1.6rem;">&#9889;</div>
                <div style="font-weight:600; color:#4ade80;">Duplex-1</div>
                <div style="color:#999; font-size:0.85rem;">Generates &rarr; Correction injected &rarr; Continues seamlessly</div>
            </div>
        </div>
        """)

        # Inputs
        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Ask a question or give an instruction...",
                    lines=2,
                )
                correction_input = gr.Textbox(
                    label="Correction (injected mid-generation)",
                    placeholder="What should the model hear while it's generating?",
                    lines=2,
                )
            with gr.Column(scale=1):
                tokens_slider = gr.Slider(
                    minimum=5, maximum=800, value=80, step=5,
                    label="Inject after N tokens",
                )
                max_tokens_slider = gr.Slider(
                    minimum=100, maximum=2048, value=2048, step=50,
                    label="Max tokens",
                )
                temp_slider = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.5, step=0.1,
                    label="Temperature",
                )

        run_btn = gr.Button("Run Side-by-Side", variant="primary", size="lg")

        # Outputs
        with gr.Row():
            with gr.Column():
                gr.HTML('<div style="font-weight:600; color:#f87171; font-size:1.05rem; margin-bottom:4px;">Qwen3-1.7B (Baseline)</div>')
                baseline_output = gr.Textbox(
                    label="",
                    lines=18,
                    interactive=False,
                    show_label=False,
                )
                baseline_status = gr.Textbox(
                    value="",
                    interactive=False,
                    show_label=False,
                    lines=1,
                    container=False,
                )

            with gr.Column():
                gr.HTML('<div style="font-weight:600; color:#4ade80; font-size:1.05rem; margin-bottom:4px;">Duplex-1.2-1.7B (Full-Duplex)</div>')
                duplex_output = gr.Textbox(
                    label="",
                    lines=18,
                    interactive=False,
                    show_label=False,
                )
                duplex_status = gr.Textbox(
                    value="",
                    interactive=False,
                    show_label=False,
                    lines=1,
                    container=False,
                )

        run_btn.click(
            fn=run_comparison,
            inputs=[prompt_input, correction_input, tokens_slider, max_tokens_slider, temp_slider],
            outputs=[baseline_output, duplex_output, baseline_status, duplex_status],
        )

        # Examples
        gr.HTML('<div style="margin-top:16px; font-weight:600; font-size:1rem;">Try these examples:</div>')
        gr.Examples(
            examples=examples,
            inputs=[prompt_input, correction_input, tokens_slider, max_tokens_slider, temp_slider],
            label="",
        )

    print(f"\nStarting demo on http://localhost:{args.port}")
    demo.launch(server_port=args.port, share=False, theme=gr.themes.Default())


if __name__ == "__main__":
    main()
