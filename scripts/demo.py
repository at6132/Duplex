"""
Investor demo: side-by-side comparison of vanilla Qwen vs Duplex-1-1.7B.

Left panel:  Vanilla Qwen3-1.7B (has to stop and restart on correction)
Right panel: Duplex-1-1.7B (updates workspace mid-stream, continues seamlessly)

Usage:
    python scripts/demo.py --duplex_ckpt checkpoints/duplex-1-1.7b/final.pt
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gradio as gr

from duplex.inference.generate import (
    load_vanilla_qwen,
    load_duplex_model,
    generate_vanilla_with_restart,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duplex_ckpt", required=True)
    parser.add_argument("--qwen_path", default="models/qwen3-1.7b-base")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Loading vanilla Qwen3-1.7B (baseline)...")
    qwen_model, qwen_tokenizer = load_vanilla_qwen(args.qwen_path)

    print("Loading Duplex-1-1.7B...")
    duplex_model = load_duplex_model(args.qwen_path, args.duplex_ckpt)

    def run_comparison(prompt, correction, tokens_before_correction, max_tokens, temperature):
        tokens_before = int(tokens_before_correction)
        max_tok = int(max_tokens)
        temp = float(temperature)

        # Baseline: generate then restart
        qwen_partial, qwen_restarted = generate_vanilla_with_restart(
            qwen_model, qwen_tokenizer,
            prompt, correction,
            tokens_before_correction=tokens_before,
            max_new_tokens=max_tok,
            temperature=temp,
        )

        # Duplex: generate with mid-stream update
        duplex_full, duplex_at_correction = duplex_model.generate_with_update(
            prompt_text=prompt,
            max_new_tokens=max_tok,
            temperature=temp,
            correction_text=correction,
            correction_after_tokens=tokens_before,
        )

        baseline_output = (
            f"--- Initial generation (before correction) ---\n"
            f"{qwen_partial}\n\n"
            f"--- [STOPPED] User correction: \"{correction}\" ---\n\n"
            f"--- Restarted from scratch with correction ---\n"
            f"{qwen_restarted}"
        )

        duplex_output = (
            f"--- Generation with live workspace update ---\n"
            f"{duplex_full}\n\n"
            f"--- Correction \"{correction}\" was injected at token {tokens_before} ---\n"
            f"--- No restart needed. Model updated workspace and continued. ---"
        )

        return baseline_output, duplex_output

    # Example prompts
    examples = [
        [
            "What is the capital of Australia? Please explain briefly.",
            "That's wrong. The capital is Canberra, not Sydney.",
            10, 150, 0.7,
        ],
        [
            "Given x = 5, compute 3*x + 7 step by step.",
            "Actually, x = 12, not 5. Please recalculate.",
            10, 150, 0.7,
        ],
        [
            "Write a brief paragraph about machine learning.",
            "Actually, I'd like you to write about quantum computing instead.",
            15, 150, 0.7,
        ],
        [
            "Create a profile for Alice: age 25, software engineer in New York.",
            "Update: Alice is 32 and works as a doctor.",
            10, 150, 0.7,
        ],
    ]

    with gr.Blocks(
        title="Duplex-1 Demo",
        theme=gr.themes.Base(),
    ) as demo:
        gr.Markdown(
            "# Duplex-1-1.7B: Full-Duplex Language Model Demo\n"
            "**Left:** Vanilla Qwen3-1.7B (must stop and restart on correction)\n\n"
            "**Right:** Duplex-1-1.7B (updates workspace mid-stream, continues seamlessly)\n\n"
            "Same base model. Same prompt. Same correction. Only difference: our full-duplex architecture."
        )

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter a prompt...",
                    lines=3,
                )
                correction_input = gr.Textbox(
                    label="Mid-stream correction",
                    placeholder="Enter correction to inject during generation...",
                    lines=2,
                )

            with gr.Column():
                tokens_slider = gr.Slider(
                    minimum=5, maximum=50, value=10, step=1,
                    label="Inject correction after N tokens",
                )
                max_tokens_slider = gr.Slider(
                    minimum=50, maximum=300, value=150, step=10,
                    label="Max tokens to generate",
                )
                temp_slider = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature",
                )

        run_btn = gr.Button("Run Comparison", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Qwen3-1.7B (Baseline)")
                baseline_output = gr.Textbox(
                    label="Output",
                    lines=15,
                    interactive=False,
                )
            with gr.Column():
                gr.Markdown("### Duplex-1-1.7B (Full-Duplex)")
                duplex_output = gr.Textbox(
                    label="Output",
                    lines=15,
                    interactive=False,
                )

        run_btn.click(
            fn=run_comparison,
            inputs=[prompt_input, correction_input, tokens_slider, max_tokens_slider, temp_slider],
            outputs=[baseline_output, duplex_output],
        )

        gr.Examples(
            examples=examples,
            inputs=[prompt_input, correction_input, tokens_slider, max_tokens_slider, temp_slider],
        )

    print(f"\nStarting demo on http://localhost:{args.port}")
    demo.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
