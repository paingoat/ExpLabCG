# Moebius student (eager — used by inference / Gradio)
from .nets.unet_lambda_prune_lite import UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8
from .nets.unet_lambda_dwconv_blocks import *  # block factories for student


def __getattr__(name):
    """Lazy-load PixelHacker teacher so Moebius-only inference does not require fla CUDA."""
    if name == "UNet2DGLAConditionModel":
        from .nets.unet_gla import UNet2DGLAConditionModel
        return UNet2DGLAConditionModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
