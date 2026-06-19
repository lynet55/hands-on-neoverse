#!/bin/bash
# Launch the interactive multi-model viewer (demos/multi_model_demo.py).
#
# Run this on a GPU node of the cluster — the Gaussian-splat rasterizer is
# CUDA-only. It loads the checkpoints under models/NeoVerse/ by default and
# prints a public *.gradio.live share link (share mode is on by default).
#
# Usage:
#   bash demos/run_multi_model_demo.sh                 # defaults
#   bash demos/run_multi_model_demo.sh --low_vram      # keep models on CPU between calls
#   bash demos/run_multi_model_demo.sh --no-share      # local link only
#
# Any extra args are forwarded straight to the Python module.
set -euo pipefail

# Run from the repo root regardless of where the script is invoked.
cd "$(dirname "$0")/.."

mkdir -p logs outputs

. /etc/profile.d/modules.sh
module add cuda/12.8

source ./neoverse/bin/activate

# share=True by default -> prints a public share link the user can open anywhere.
python -u -m demos.multi_model_demo "$@"
