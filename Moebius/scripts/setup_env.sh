#!/usr/bin/env bash
# Create conda env and install Moebius deps for Ubuntu + RTX 5090 (sm_120).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-moebius}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FLA_VERSION="${FLA_VERSION:-0.3.2}"
RECREATE="${RECREATE:-0}"

echo "[setup_env] Repo root: ${REPO_ROOT}"
echo "[setup_env] Conda env: ${ENV_NAME} (python ${PYTHON_VERSION})"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup_env] ERROR: conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Newer conda rejects/ignores bare -y on some commands; use ALWAYS_YES instead.
export CONDA_ALWAYS_YES=true

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${RECREATE}" == "1" ]]; then
    echo "[setup_env] Removing existing env ${ENV_NAME} (RECREATE=1)..."
    conda env remove -n "${ENV_NAME}"
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  else
    echo "[setup_env] Env ${ENV_NAME} already exists — skipping create (set RECREATE=1 to recreate)."
  fi
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

echo "[setup_env] Installing torch==${TORCH_VERSION}+cu128 / torchvision==${TORCHVISION_VERSION}..."
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"

echo "[setup_env] Installing requirements.txt..."
python -m pip install -r "${REPO_ROOT}/requirements.txt"

echo "[setup_env] Installing flash-linear-attention[cuda]==${FLA_VERSION} (optional for Moebius-only infer)..."
if ! python -m pip install "flash-linear-attention[cuda]==${FLA_VERSION}"; then
  echo "[setup_env] WARN: flash-linear-attention==${FLA_VERSION} failed; trying latest compatible build..."
  if ! python -m pip install "flash-linear-attention[cuda]"; then
    echo "[setup_env] WARN: flash-linear-attention install failed."
    echo "         Moebius Gradio / student inference can still run (GLA is lazy-imported)."
    echo "         PixelHacker teacher training will need a working FLA build."
  fi
fi

echo "[setup_env] Verifying torch / CUDA..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("arch list:", torch.cuda.get_arch_list())
    archs = torch.cuda.get_arch_list()
    if not any("sm_120" in a or "120" in a for a in archs):
        print("WARNING: sm_120 not in arch list — RTX 5090 may need a cu128+ wheel.")
else:
    print("WARNING: CUDA not available in this environment.")
PY

echo "[setup_env] Done. Activate with: conda activate ${ENV_NAME}"
echo "[setup_env] Next: bash scripts/download_model.sh"
