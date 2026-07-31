#!/usr/bin/env python3
"""Gradio demo: upload an image, draw a mask, run Moebius inpainting."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import gradio as gr
import torch

from infer.gradio_utils import editor_to_image_and_mask, make_mask_overlay
from infer.utils import build_pipeline


CHECKPOINT_VARIANTS = {
    "ft_celebahq": "weight/Moebius/ft_celebahq/diffusion_pytorch_model.bin",
    "ft_places2": "weight/Moebius/ft_places2/diffusion_pytorch_model.bin",
    "ft_ffhq": "weight/Moebius/ft_ffhq/diffusion_pytorch_model.bin",
    "pretrained": "weight/Moebius/pretrained/diffusion_pytorch_model.bin",
}


class PipelineHolder:
    """Lazy-load / reload Moebius removal pipeline when checkpoint changes."""

    def __init__(self, model_config: str, device: str, default_weight: str):
        self.model_config = model_config
        self.device = device
        self.weight_path = default_weight
        self.pipe = None

    def ensure(self, weight_path: Optional[str] = None):
        target = weight_path or self.weight_path
        if self.pipe is not None and target == self.weight_path:
            return self.pipe

        if not Path(target).is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {target}\n"
                "Run: bash scripts/download_model.sh"
            )
        vae_cfg = Path("weight/vae/config.json")
        if not vae_cfg.is_file():
            raise FileNotFoundError(
                "VAE not found under weight/vae/. Run: bash scripts/download_model.sh"
            )

        # Free previous model before loading another variant
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        args = SimpleNamespace(
            model_config=self.model_config,
            model_weight=target,
            device=self.device,
        )
        self.pipe = build_pipeline(args)
        self.weight_path = target
        return self.pipe


def parse_args():
    parser = argparse.ArgumentParser(description="Moebius Gradio inpainting demo")
    parser.add_argument(
        "--model-config",
        type=str,
        default="config/model_cfg/moebius.yaml",
    )
    parser.add_argument(
        "--model-weight",
        type=str,
        default=CHECKPOINT_VARIANTS["ft_celebahq"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a public Gradio share link (default: True). Use --no-share to disable.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./outputs/gradio",
        help="Directory to save Gradio results",
    )
    return parser.parse_args()


def build_demo(holder: PipelineHolder, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    def run_inpaint(
        editor_value,
        checkpoint_name,
        num_steps,
        cfg,
        paste,
        compensate,
        noise_offset,
        resolution,
        seed,
    ):
        try:
            image, mask = editor_to_image_and_mask(editor_value)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        weight_path = CHECKPOINT_VARIANTS.get(checkpoint_name, holder.weight_path)
        try:
            pipe = holder.ensure(weight_path)
        except FileNotFoundError as exc:
            raise gr.Error(str(exc)) from exc

        # Pipeline maps retry -> torch seed (0 if retry == 0 else retry)
        results = pipe(
            [image],
            [mask],
            image_size=int(resolution),
            num_steps=int(num_steps),
            guidance_scale=float(cfg),
            paste=bool(paste),
            compensate=bool(compensate),
            noise_offset=float(noise_offset),
            mute=False,
            retry=int(seed),
        )
        result = results[0]
        overlay = make_mask_overlay(image, mask)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = save_dir / f"inpaint_{stamp}.png"
        result.save(out_path)

        return overlay, mask, result, str(out_path)

    # Preload default checkpoint so first click is faster (fail soft if missing)
    try:
        holder.ensure()
        status = f"Loaded: {holder.weight_path}"
    except FileNotFoundError as exc:
        status = f"Weights not loaded yet — {exc}"

    with gr.Blocks(title="Moebius Inpainting") as demo:
        gr.Markdown(
            "# Moebius Inpainting\n"
            "Upload an image, **brush the region to fill**, then run inpainting. "
            "White / painted strokes mark the hole (same convention as the CLI)."
        )
        gr.Markdown(f"**Status:** {status}")

        with gr.Row():
            with gr.Column(scale=1):
                editor = gr.ImageEditor(
                    label="Image + mask brush",
                    type="pil",
                    image_mode="RGBA",
                    brush=gr.Brush(default_size=30, colors=["#ffffff"], color_mode="fixed"),
                    layers=True,
                    sources=["upload", "clipboard"],
                )
                checkpoint = gr.Dropdown(
                    label="Checkpoint",
                    choices=list(CHECKPOINT_VARIANTS.keys()),
                    value="ft_celebahq",
                )
                with gr.Row():
                    num_steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                    cfg = gr.Slider(1.0, 7.5, value=2.5, step=0.1, label="CFG")
                with gr.Row():
                    resolution = gr.Slider(256, 1024, value=512, step=64, label="Resolution (short side)")
                    seed = gr.Number(value=0, precision=0, label="Seed (0 = default)")
                with gr.Row():
                    paste = gr.Checkbox(value=True, label="Paste")
                    compensate = gr.Checkbox(value=False, label="Compensate")
                noise_offset = gr.Number(value=0.0357, label="Noise offset")
                run_btn = gr.Button("Run inpainting", variant="primary")

            with gr.Column(scale=1):
                out_overlay = gr.Image(label="Mask overlay", type="pil")
                out_mask = gr.Image(label="Binary mask", type="pil", image_mode="L")
                out_result = gr.Image(label="Inpainted result", type="pil")
                out_path = gr.Textbox(label="Saved to", interactive=False)

        run_btn.click(
            fn=run_inpaint,
            inputs=[
                editor,
                checkpoint,
                num_steps,
                cfg,
                paste,
                compensate,
                noise_offset,
                resolution,
                seed,
            ],
            outputs=[out_overlay, out_mask, out_result, out_path],
        )

    return demo


def main():
    # Optional .env for HF cache paths (weights are local; still useful)
    try:
        from dotenv import load_dotenv
        env_path = Path(".env")
        if env_path.is_file():
            load_dotenv(env_path)
        else:
            example = Path(".env.example")
            if example.is_file():
                load_dotenv(example)
    except ImportError:
        pass

    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA not available; falling back to CPU.")
        args.device = "cpu"

    holder = PipelineHolder(
        model_config=args.model_config,
        device=args.device,
        default_weight=args.model_weight,
    )
    demo = build_demo(holder, Path(args.save_dir))
    demo.queue(max_size=8).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
