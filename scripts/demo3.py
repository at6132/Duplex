"""
Demo 3: Breakthrough showcase — the clearest way to show why Duplex is a leap forward.

Single scenario, streaming, with "problem vs solution" framing and strong visuals.
Usage: python scripts/demo3.py --duplex_ckpt checkpoints/duplex-1.1-1.7b/final.pt
"""

import argparse
import sys
import threading
import time
from queue import Queue, Empty

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duplex_ckpt", required=True)
    parser.add_argument("--qwen_path", default="models/qwen3-1.7b-base")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()

    print("Loading models...")
    qwen_model, qwen_tokenizer = load_vanilla_qwen(args.qwen_path, quantize=False)
    duplex_model = load_duplex_model(args.qwen_path, args.duplex_ckpt)
    print("Ready.\n")

    # Preset scenarios tuned so injection always fires (inject early) and story is clear
    SCENARIOS = [
        {
            "name": "Fact correction",
            "prompt": "What is the capital of Australia? Explain in two or three sentences.",
            "correction": "That's wrong — the capital is Canberra, not Sydney.",
            "inject_after": 12,
            "max_tokens": 120,
            "temp": 0.5,
        },
        {
            "name": "Change of task",
            "prompt": "Write a short Python function to compute the factorial of n. Add a docstring.",
            "correction": "Actually use a loop instead of recursion, and add type hints.",
            "inject_after": 15,
            "max_tokens": 180,
            "temp": 0.4,
        },
        {
            "name": "Topic switch",
            "prompt": "Explain how photosynthesis works in plants in a few sentences.",
            "correction": "Switch to explaining cellular respiration instead.",
            "inject_after": 18,
            "max_tokens": 200,
            "temp": 0.5,
        },
    ]

    def run_demo(scenario_name: str):
        s = next((x for x in SCENARIOS if x["name"] == scenario_name), SCENARIOS[0])
        prompt = SYSTEM_PROMPT + s["prompt"]
        correction = s["correction"]
        inject = s["inject_after"]
        max_tok = s["max_tokens"]
        temp = s["temp"]

        state = {"baseline": "", "duplex": "", "done_b": False, "done_d": False}
        times = {"baseline": 0.0, "duplex": 0.0}
        queue = Queue()

        def run_baseline():
            t0 = time.perf_counter()
            try:
                for text in generate_vanilla_with_restart_streaming(
                    qwen_model, qwen_tokenizer,
                    prompt, correction,
                    tokens_before_correction=inject,
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
                    prompt_text=prompt,
                    max_new_tokens=max_tok,
                    temperature=temp,
                    correction_text=correction,
                    correction_after_tokens=inject,
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
                "Running..." if not state["done_b"] else f"Done in {times['baseline']:.1f}s",
                "Running..." if not state["done_d"] else f"Done in {times['duplex']:.1f}s",
            )

        yield (
            state["baseline"],
            state["duplex"],
            f"Done in {times['baseline']:.1f}s",
            f"Done in {times['duplex']:.1f}s",
        )

    with gr.Blocks(
        title="Duplex — The Breakthrough",
        css="""
        .hero { text-align: center; padding: 28px 20px 20px; }
        .hero h1 { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.02em; }
        .hero .tagline { font-size: 1.1rem; color: #94a3b8; margin-top: 8px; max-width: 560px; margin-left: auto; margin-right: auto; }
        .problem-box { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px 20px; margin: 8px 0; }
        .problem-box h3 { margin: 0 0 6px 0; font-size: 1rem; color: #f87171; }
        .problem-box p { margin: 0; color: #94a3b8; font-size: 0.9rem; line-height: 1.45; }
        .breakthrough-box { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px 20px; margin: 8px 0; }
        .breakthrough-box h3 { margin: 0 0 6px 0; font-size: 1rem; color: #4ade80; }
        .breakthrough-box p { margin: 0; color: #94a3b8; font-size: 0.9rem; line-height: 1.45; }
        .col-label { font-weight: 700; font-size: 1.05rem; margin-bottom: 8px; }
        .col-label.baseline { color: #f87171; }
        .col-label.duplex { color: #4ade80; }
        .scenario-card { border: 1px solid #334155; border-radius: 10px; padding: 12px 16px; margin: 6px 0; cursor: pointer; }
        .scenario-card:hover { background: #1e293b; }
        """
    ) as demo:

        # Hero
        gr.HTML("""
        <div class="hero">
            <h1>Duplex: The first LLM that accepts input while it's talking</h1>
            <p class="tagline">Every other model must stop, restart, and lose context. Duplex updates in real time and keeps going.</p>
        </div>
        """)

        # Problem / Breakthrough
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="problem-box">
                    <h3>Today’s standard</h3>
                    <p>If you correct an LLM mid-response, it has to <strong>stop</strong>, discard what it said, and <strong>restart from scratch</strong> with your correction. Wasted tokens, broken flow, no true dialogue.</p>
                </div>
                """)
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="breakthrough-box">
                    <h3>The breakthrough</h3>
                    <p>Duplex <strong>receives your correction while it’s still generating</strong>. It updates an internal workspace and continues—no stop, no restart. Full-duplex conversation.</p>
                </div>
                """)

        gr.HTML("""<div style="text-align:center; padding: 16px 0 8px 0;"><strong>See it live</strong> — same prompt, same correction. Left: standard model. Right: Duplex.</div>""")

        # Scenario choice
        scenario_dropdown = gr.Dropdown(
            choices=[s["name"] for s in SCENARIOS],
            value=SCENARIOS[0]["name"],
            label="Scenario",
        )
        run_btn = gr.Button("Run demo", variant="primary", size="lg")

        # Show what will happen
        gr.HTML("""
        <div style="font-size:0.85rem; color:#64748b; margin-bottom:12px;">
            Prompt and correction are fixed per scenario. The model starts answering; after a few tokens we inject the correction. Watch the left side stop and restart; the right side shows the injection and continues.
        </div>
        """)

        # Outputs
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="col-label baseline">Standard LLM — stops and restarts</div>')
                baseline_out = gr.Textbox(lines=14, show_label=False, interactive=False)
                baseline_status = gr.Textbox(show_label=False, interactive=False, container=False)
            with gr.Column():
                gr.HTML('<div class="col-label duplex">Duplex — receives injection and continues</div>')
                duplex_out = gr.Textbox(lines=14, show_label=False, interactive=False)
                duplex_status = gr.Textbox(show_label=False, interactive=False, container=False)

        run_btn.click(
            fn=run_demo,
            inputs=[scenario_dropdown],
            outputs=[baseline_out, duplex_out, baseline_status, duplex_status],
        )

        # Footer
        gr.HTML("""
        <div style="text-align:center; padding: 24px 20px; color: #64748b; font-size: 0.9rem;">
            Same base model (Qwen3-1.7B). Same prompt. Same correction. Only difference: Duplex’s full-duplex architecture.
        </div>
        """)

    print(f"Breakthrough demo: http://localhost:{args.port}")
    demo.launch(server_port=args.port, share=False, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
