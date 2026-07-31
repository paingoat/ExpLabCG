# Moebius Setup Guide (Ubuntu + RTX 5090)

End-to-end lab setup for Moebius inference and the Gradio inpainting demo.
Weights and Hugging Face caches default to `/backup/data/art-gen` via `.env.example`.

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Ubuntu (lab) | Scripts are bash; not for Windows |
| NVIDIA driver | CUDA **12.8+** capability (driver **570+** recommended for RTX 5090 / Blackwell) |
| Conda | Miniconda or Anaconda (`conda` on `PATH`) |
| Disk | Space under `/backup/data/art-gen` for HF caches + `./weight` checkpoints |
| Network | Access to Hugging Face Hub (`hustvl/Moebius`, `hustvl/PixelHacker`) |

Check the GPU:

```bash
nvidia-smi
```

You should see an RTX 5090 (or other Blackwell card). The system CUDA toolkit version does **not** need to match the PyTorch wheel exactly — PyTorch ships its own CUDA runtime.

## 2. Repository and environment files

```bash
cd /path/to/Moebius
cp .env.example .env
# Edit .env only if your HF cache root differs from /backup/data/art-gen
```

`.env` keys:

```bash
HF_HOME=/backup/data/art-gen
HF_HUB_CACHE=/backup/data/art-gen/hub
TRANSFORMERS_CACHE=/backup/data/art-gen/transformers
```

## 3. Create conda env and install dependencies

```bash
chmod +x scripts/*.sh
bash scripts/setup_env.sh
```

What this does:

1. Creates conda env **`moebius`** with **Python 3.11** (non-interactive via `CONDA_ALWAYS_YES`, no `-y`)
2. Installs **torch 2.7.1 + torchvision 0.22.1** from the official **cu128** index (includes Blackwell `sm_120`)
3. Installs `requirements.txt` (diffusers 0.38.0, Gradio, etc.)
4. Tries `flash-linear-attention[cuda]==0.3.2` (needed for PixelHacker teacher training; **optional** for Moebius-only Gradio)

Optional knobs:

```bash
RECREATE=1 bash scripts/setup_env.sh          # wipe and recreate env
ENV_NAME=moebius PYTHON_VERSION=3.11 bash scripts/setup_env.sh
```

Activate:

```bash
conda activate moebius
```

Verify Blackwell support:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_arch_list())"
```

Expect `sm_120` (or `compute_120`) in the arch list. If it is missing, you installed a CUDA ≤12.6 wheel — re-run `setup_env.sh`.

## 4. Download model checkpoints

```bash
bash scripts/download_model.sh
# Force re-download:
bash scripts/download_model.sh --force
```

Target layout (same as README):

```text
weight/
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── Moebius/
    ├── pretrained/diffusion_pytorch_model.bin
    ├── ft_places2/diffusion_pytorch_model.bin
    ├── ft_celebahq/diffusion_pytorch_model.bin
    └── ft_ffhq/diffusion_pytorch_model.bin
```

Sources:

- VAE: [hustvl/PixelHacker](https://huggingface.co/hustvl/PixelHacker) (`vae/`)
- UNet: [hustvl/Moebius](https://huggingface.co/hustvl/Moebius)

Downloads use `HF_HOME` / `HF_HUB_CACHE` from `.env`, then copy into `./weight`.

## 5. CLI smoke test (optional)

Place paired images and masks under folders (same sort order), then:

```bash
conda activate moebius
python -m infer.infer_moebius \
    --model-config config/model_cfg/moebius.yaml \
    --model-weight weight/Moebius/ft_celebahq/diffusion_pytorch_model.bin \
    --real-dir data/images \
    --mask-dir data/masks \
    --save-dir ./outputs \
    --cfg 2.0 \
    --batch-size 1 \
    --num-workers 2
```

Defaults that match the original repo numerics: `num_steps=20`, `cfg=2.5` (CLI default; README example uses 2.0), `paste=True`, `compensate=False`, `noise_offset=0.0357`, fp32 pipeline dtype.

## 6. Gradio demo (draw mask on image)

```bash
bash scripts/run_gradio.sh
```

Or:

```bash
conda activate moebius
python app_gradio.py --server-name 0.0.0.0 --server-port 7860
```

Open `http://<lab-host>:7860`.

Usage:

1. Upload a real image
2. Brush the region to inpaint (white strokes = hole)
3. Choose checkpoint (`ft_celebahq` default, `ft_places2`, `ft_ffhq`, `pretrained`)
4. Adjust steps / CFG / paste / compensate / noise offset / seed if needed
5. Click **Run inpainting** — results also save under `./outputs/gradio/`

Environment overrides for `run_gradio.sh`:

```bash
SERVER_PORT=7861 MODEL_WEIGHT=weight/Moebius/ft_places2/diffusion_pytorch_model.bin bash scripts/run_gradio.sh
```

Preflight checks (script exits early on failure):

- conda env `moebius` exists
- `torch` importable; warns if CUDA / `sm_120` missing
- VAE + selected Moebius checkpoint files exist
- `gradio` / `diffusers` installed

## 7. Troubleshooting

### `CondaToSNonInteractiveError` / Terms of Service

Newer Conda blocks non-interactive `conda create` until default Anaconda channel ToS are accepted. `scripts/setup_env.sh` runs `conda tos accept` automatically. Manual fix:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
bash scripts/setup_env.sh
```

### `sm_120` not compatible / no kernel image

Wrong PyTorch CUDA build. Uninstall and reinstall cu128:

```bash
conda activate moebius
pip uninstall -y torch torchvision torchaudio
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

### `flash-linear-attention` build fails

Safe for **Moebius Gradio / student inference** — GLA (PixelHacker teacher) is lazy-imported. Only teacher distillation needs FLA. Retry later with a newer `flash-linear-attention[cuda]` after torch is confirmed working.

### Missing weights

```bash
bash scripts/download_model.sh --force
ls -R weight/
```

### Hugging Face auth / rate limits

Public repos normally need no token. If rate-limited:

```bash
huggingface-cli login
# or set HF_TOKEN in .env
```

### Gradio “Please draw a mask”

Brush the hole region after upload. Painted strokes become the white inpaint mask (same convention as CLI masks: white ≥128 = fill region).

### OOM when switching checkpoints

The app reloads weights on checkpoint change and calls `torch.cuda.empty_cache()`. If VRAM is tight, restart Gradio with a single `--model-weight`.

## 8. Stack summary (reproducibility)

| Component | Version / choice |
|---|---|
| Python | 3.11 |
| torch | 2.7.1+cu128 |
| torchvision | 0.22.1 |
| diffusers | 0.38.0 |
| transformers | 4.56.2 |
| accelerate | 1.14.0 |
| flash-linear-attention | 0.3.2 (best-effort) |
| Gradio | ≥5,<6 |
| Infer dtype | float32 (unchanged from upstream) |
| Default Gradio checkpoint | `ft_celebahq` |

Inference logic intentionally reuses `infer.utils.build_pipeline` and `RemovalSDXLPipeline_BatchMode` so sampling behavior stays aligned with the original CLI.
