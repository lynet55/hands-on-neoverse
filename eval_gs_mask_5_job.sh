#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=gs_mask_eval5
#SBATCH --partition=jobs
#SBATCH --time=0-01:00:00
#SBATCH --chdir=/work/courses/3dv/team32/handy-NeoVerse
#SBATCH --output=/work/courses/3dv/team32/handy-NeoVerse/logs/eval_gs_mask5_%j.out
#SBATCH --error=/work/courses/3dv/team32/handy-NeoVerse/logs/eval_gs_mask5_%j.err

mkdir -p logs outputs

. /etc/profile.d/modules.sh
module add cuda/12.8

source ./neoverse/bin/activate

python -u eval_segmentation.py \
    --npz \
        diffsynth/data/training_data/clip-001053.npz \
        diffsynth/data/training_data/clip-001068.npz \
        diffsynth/data/training_data/clip-001083.npz \
        diffsynth/data/training_data/clip-001100.npz \
        diffsynth/data/training_data/clip-001120.npz \
    --gs_mask_path models/NeoVerse/gs_mask_model_latest.ckpt \
    --output outputs/eval_gs_mask_latest.mp4 \
    --stride 3 \
    --batch_size 4
