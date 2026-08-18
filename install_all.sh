#!/usr/bin/env bash
#
# Install the full Oneliner host model toolchain with pinned versions.
# Run this inside the Python/conda environment you build with, e.g.:
#
#     conda activate ariel_ml
#     ./install_all.sh
#
# See docs/INSTALLATION.md for details.

set -euo pipefail

python - <<'EOF'
import sys

if sys.version_info < (3, 12):
    sys.exit(f"error: Python 3.12 or newer is required (found {sys.version.split()[0]})")
EOF

echo "==> Installing compiler and model import packages (pinned)..."
pip install \
    "iree-base-compiler[onnx]==3.11.0" \
    "tosa-converter-for-tflite==2026.2.0" \
    "tensorflow==2.21.0" \
    "iree-tools-tf==20250718.1326" \
    "iree-turbine==3.9.0"

echo "==> Installing the PyTorch CPU build..."
pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cpu

echo "==> Verifying installation..."
iree-compile --version
tosa-converter-for-tflite --version
iree-import-onnx --help >/dev/null
iree-import-tf --help >/dev/null
python -c "import torch, iree.turbine.aot"
python -c "import tensorflow"

echo "==> All host tools installed successfully."
