#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=neoverse_multi_model_demo
#SBATCH --partition=jobs
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --chdir=/work/courses/3dv/team32/hands-on-neoverse
#SBATCH --output=/work/courses/3dv/team32/hands-on-neoverse/logs/multi_model_demo_%j.out
#SBATCH --error=/work/courses/3dv/team32/hands-on-neoverse/logs/multi_model_demo_%j.err
#
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
#   sbatch demos/run_multi_model_demo.sh                # submit as a cluster job
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
