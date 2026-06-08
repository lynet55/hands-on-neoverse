#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=gsparam_ablation
#SBATCH --partition=jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=0-01:00:00
#SBATCH --chdir=/work/courses/3dv/team32/hands-on-neoverse
#SBATCH --output=/work/courses/3dv/team32/hands-on-neoverse/logs/gsparam_ablation_%j.out
#SBATCH --error=/work/courses/3dv/team32/hands-on-neoverse/logs/gsparam_ablation_%j.err

# Eval-only ablation for the Gaussian Params variant (#1 background fix, #2 multi-frame).
# Each config is run over the SAME fixed clip set so the gsparam mIoU deltas are
# directly comparable; the Naive Gaussian numbers should be identical everywhere.
mkdir -p logs outputs/gsparam_ablation

. /etc/profile.d/modules.sh
module add cuda/12.8

set -eo pipefail
source ./neoverse/bin/activate
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi || true

CLIPS="/work/courses/3dv/team32/training_data_modal/clip-001100.npz \
       /work/courses/3dv/team32/training_data_modal/clip-001270.npz \
       /work/courses/3dv/team32/training_data_modal/clip-001284.npz"
CKPT=models/NeoVerse/gs_mask_model_best.ckpt

run_cfg () {
    name="$1"; shift
    echo ""
    echo "########## CONFIG: ${name} ##########"
    python -u eval_segmentation_gs_mask.py \
        --npz ${CLIPS} \
        --gs_mask_head_path ${CKPT} \
        --output outputs/gsparam_ablation/${name}/eval.mp4 \
        "$@"
}

run_cfg baseline
run_cfg bg_bias            --bg_bias 5.0
run_cfg bg_alpha_thresh    --bg_alpha_thresh 0.5
run_cfg multiframe8        --gsparam_seq_len 8
run_cfg multiframe8_alpha  --gsparam_seq_len 8 --bg_alpha_thresh 0.5

echo ""
echo "ABLATION DONE."
