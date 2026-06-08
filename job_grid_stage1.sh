#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=grid_stage1_neoverse
#SBATCH --partition=jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=0-00:30:00
#SBATCH --chdir=/work/courses/3dv/team32/hands-on-neoverse
#SBATCH --output=/work/courses/3dv/team32/hands-on-neoverse/logs/grid_stage1_%j.out
#SBATCH --error=/work/courses/3dv/team32/hands-on-neoverse/logs/grid_stage1_%j.err

# Stage 1: dump Naive Gaussian + Gaussian Params + GT overlays for the keyframes.
mkdir -p logs

. /etc/profile.d/modules.sh
module add cuda/12.8

set -eo pipefail
source ./neoverse/bin/activate
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi || true

python -u make_grid_neoverse.py \
    --npz /work/courses/3dv/team32/training_data_modal/clip-001100.npz \
    --stream stream1201-1 \
    --frames 39 42 45 48 \
    --out_dir outputs/grid_clip-001100

echo "Stage 1 done."
