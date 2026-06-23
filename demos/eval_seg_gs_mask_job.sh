#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=neoverse_eval_seg_gs_mask
#SBATCH --partition=jobs
#SBATCH --time=0-01:00:00
#SBATCH --chdir=/work/courses/3dv/team32/hands-on-neoverse
#SBATCH --output=/work/courses/3dv/team32/hands-on-neoverse/logs/eval_seg_gs_mask_%j.out
#SBATCH --error=/work/courses/3dv/team32/hands-on-neoverse/logs/eval_seg_gs_mask_%j.err

mkdir -p logs outputs

. /etc/profile.d/modules.sh
module add cuda/12.8

source ./neoverse/bin/activate

python -u -m demos.eval_segmentation_gs_mask \
    --npz \
        demo_data/clip-000088.npz \
        demo_data/clip-000089.npz \
        demo_data/clip-000090.npz \
        demo_data/clip-000091.npz \
        demo_data/clip-000092.npz \
    --reconstructor_path models/NeoVerse/reconstructor.ckpt \
    --gs_mask_head_path models/NeoVerse/gs_mask_model_run20260510-175056_epoch006.ckpt \
    --output outputs/eval_seg_gs_mask.mp4
