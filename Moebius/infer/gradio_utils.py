"""Helpers for converting Gradio ImageEditor output into image + binary mask."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
from PIL import Image


def _to_pil_rgb(img: Any) -> Image.Image:
    if img is None:
        raise ValueError("Expected an image, got None.")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.asarray(img)
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    if arr.shape[-1] == 4:
        return Image.fromarray(arr.astype(np.uint8), mode="RGBA").convert("RGB")
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _layers_to_mask(layers: list, size: Tuple[int, int]) -> Optional[Image.Image]:
    """Build a binary mask from ImageEditor layer alpha / non-zero ink."""
    if not layers:
        return None

    w, h = size
    acc = np.zeros((h, w), dtype=np.uint8)

    for layer in layers:
        if layer is None:
            continue
        if isinstance(layer, Image.Image):
            layer_img = layer
        else:
            arr = np.asarray(layer)
            if arr.ndim == 2:
                layer_img = Image.fromarray(arr.astype(np.uint8), mode="L")
            elif arr.shape[-1] == 4:
                layer_img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
            else:
                layer_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")

        if layer_img.size != (w, h):
            layer_img = layer_img.resize((w, h), Image.Resampling.NEAREST)

        if layer_img.mode == "RGBA":
            alpha = np.asarray(layer_img.split()[-1])
            acc = np.maximum(acc, (alpha >= 8).astype(np.uint8) * 255)
        else:
            gray = np.asarray(layer_img.convert("L"))
            acc = np.maximum(acc, (gray >= 8).astype(np.uint8) * 255)

    if acc.max() == 0:
        return None
    return Image.fromarray(acc, mode="L")


def _composite_diff_mask(background: Image.Image, composite: Image.Image) -> Optional[Image.Image]:
    """Fallback: mark pixels that differ from the background as mask."""
    bg = np.asarray(background.convert("RGB"), dtype=np.int16)
    cp = np.asarray(composite.convert("RGB"), dtype=np.int16)
    if bg.shape != cp.shape:
        composite = composite.resize(background.size, Image.Resampling.NEAREST)
        cp = np.asarray(composite.convert("RGB"), dtype=np.int16)
    diff = np.abs(bg - cp).sum(axis=-1)
    mask = (diff >= 12).astype(np.uint8) * 255
    if mask.max() == 0:
        return None
    return Image.fromarray(mask, mode="L")


def editor_to_image_and_mask(
    editor_value: Union[dict, Image.Image, np.ndarray, None],
) -> Tuple[Image.Image, Image.Image]:
    """
    Convert a Gradio ImageEditor value (or plain image) into (RGB image, L mask).

    Mask convention matches the Moebius CLI: white (>=128) = region to inpaint.
    Painted brush strokes become the white hole region.
    """
    if editor_value is None:
        raise ValueError("Please upload an image and draw a mask.")

    # Plain image upload without editor dict — cannot infer a mask
    if isinstance(editor_value, (Image.Image, np.ndarray)):
        raise ValueError("Please draw a mask on the image (brush the region to inpaint).")

    if not isinstance(editor_value, dict):
        raise ValueError(f"Unsupported editor value type: {type(editor_value)}")

    background = editor_value.get("background")
    layers = editor_value.get("layers") or []
    composite = editor_value.get("composite")

    if background is None and composite is not None:
        background = composite
    if background is None:
        raise ValueError("Please upload an image first.")

    image = _to_pil_rgb(background)
    mask = _layers_to_mask(layers, image.size)

    if mask is None and composite is not None:
        mask = _composite_diff_mask(image, _to_pil_rgb(composite))

    if mask is None:
        raise ValueError("Please draw a mask on the image (brush the region to inpaint).")

    # Binarize to 0 / 255 for pipeline preprocess
    mask_arr = np.asarray(mask.convert("L"))
    mask_bin = np.where(mask_arr >= 128, 255, 0).astype(np.uint8)
    if mask_bin.max() == 0:
        raise ValueError("Mask is empty. Please brush the region to inpaint.")

    return image, Image.fromarray(mask_bin, mode="L")


def ensure_square_pair(
    image: Image.Image,
    mask: Image.Image,
    resolution: int,
) -> Tuple[Image.Image, Image.Image]:
    """
    Match CLI SimpleInferDataset: force square resolution x resolution.
    Moebius lambda layers assume square feature maps (h = w = int(sqrt(n))).
    """
    resolution = int(resolution)
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    # Keep VAE / UNet friendly sizes (multiples of 64)
    resolution = max(64, (resolution // 64) * 64)

    image = image.convert("RGB")
    mask = mask.convert("L")
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
    if mask.size != (resolution, resolution):
        mask = mask.resize((resolution, resolution), Image.Resampling.NEAREST)

    mask_arr = np.where(np.asarray(mask) >= 128, 255, 0).astype(np.uint8)
    return image, Image.fromarray(mask_arr, mode="L")


def make_mask_overlay(image: Image.Image, mask: Image.Image, color=(255, 64, 64), alpha=0.45) -> Image.Image:
    """RGB visualization of the mask overlaid on the input image."""
    base = image.convert("RGBA")
    m = np.asarray(mask.convert("L").resize(image.size, Image.Resampling.NEAREST))
    overlay = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = (m >= 128).astype(np.uint8) * int(255 * alpha)
    return Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA")).convert("RGB")
