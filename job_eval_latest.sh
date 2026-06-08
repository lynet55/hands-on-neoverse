#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=eval_ft_latest
#SBATCH --partition=jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=0-00:30:00
#SBATCH --chdir=/work/courses/3dv/team32/hands-on-neoverse
#SBATCH --output=/work/courses/3dv/team32/hands-on-neoverse/logs/eval_ft_latest_%j.out
#SBATCH --error=/work/courses/3dv/team32/hands-on-neoverse/logs/eval_ft_latest_%j.err
. /etc/profile.d/modules.sh; module add cuda/12.8
set -eo pipefail; source ./neoverse/bin/activate
CLIPS="/work/courses/3dv/team32/training_data_modal/clip-001100.npz /work/courses/3dv/team32/training_data_modal/clip-001270.npz /work/courses/3dv/team32/training_data_modal/clip-001284.npz"
echo "=== _latest (step 1500) ==="
python -u eval_segmentation_gs_mask.py --npz ${CLIPS} --gs_mask_head_path models/NeoVerse/gs_mask_lovasz_detach_latest.ckpt --output outputs/ft_latest/eval.mp4
echo "EVAL LATEST DONE."
