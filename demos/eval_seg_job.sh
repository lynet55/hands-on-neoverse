#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=neoverse_eval_seg
#SBATCH --partition=jobs
#SBATCH --time=0-01:00:00
#SBATCH --chdir=/work/courses/3dv/team32/hands-on-neoverse
#SBATCH --output=/work/courses/3dv/team32/hands-on-neoverse/logs/eval_seg_%j.out
#SBATCH --error=/work/courses/3dv/team32/hands-on-neoverse/logs/eval_seg_%j.err

mkdir -p logs outputs

. /etc/profile.d/modules.sh
module add cuda/12.8

source ./neoverse/bin/activate

python -u -m demos.eval_segmentation \
    --npz \
        demo_data/clip-000088.npz \
        demo_data/clip-000089.npz \
        demo_data/clip-000090.npz \
        demo_data/clip-000091.npz \
        demo_data/clip-000092.npz \
    --hand_head_path models/NeoVerse/hand_seg_model_opt_latest.ckpt \
    --output outputs/eval_seg.mp4
