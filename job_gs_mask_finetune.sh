#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=gs_mask_finetune
#SBATCH --partition=jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=0-04:00:00
#SBATCH --chdir=/work/courses/3dv/team32/hands-on-neoverse
#SBATCH --output=/work/courses/3dv/team32/hands-on-neoverse/logs/gs_mask_finetune_%j.out
#SBATCH --error=/work/courses/3dv/team32/hands-on-neoverse/logs/gs_mask_finetune_%j.err

# Bounded fine-tune of the Gaussian-Params head from the existing best checkpoint,
# adding Lovász-softmax (#3) + geometry-detach (#6). Restarts the LR/step schedule,
# stops at GSMASK_MAX_STEPS, then re-evaluates the new checkpoint vs baseline.
mkdir -p logs
. /etc/profile.d/modules.sh
module add cuda/12.8
set -eo pipefail
source ./neoverse/bin/activate
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi || true

NEW_PREFIX=models/NeoVerse/gs_mask_lovasz_detach

# ---- bounded fine-tune ----
export GSMASK_DATA_ROOT=/work/courses/3dv/team32/training_data_modal
export GSMASK_RESUME_FROM=models/NeoVerse/gs_mask_model_best.ckpt
export GSMASK_FINETUNE=1
export GSMASK_RUN_ID=lovasz_detach_probe
export GSMASK_SAVE_PREFIX=${NEW_PREFIX}
export GSMASK_LOVASZ=0.5
export GSMASK_DETACH_GEOM=1
export GSMASK_LR=5e-5
export GSMASK_EPOCHS=50
export GSMASK_MAX_STEPS=1500
export GSMASK_SAVE_EVERY=250

python -u -m diffsynth.data.training.training_gs_mask

echo ""
echo "########## RE-EVAL: fine-tuned (Lovász+detach) vs baseline ##########"
CLIPS="/work/courses/3dv/team32/training_data_modal/clip-001100.npz \
       /work/courses/3dv/team32/training_data_modal/clip-001270.npz \
       /work/courses/3dv/team32/training_data_modal/clip-001284.npz"

echo "=== NEW checkpoint (${NEW_PREFIX}_best.ckpt) ==="
python -u eval_segmentation_gs_mask.py --npz ${CLIPS} \
    --gs_mask_head_path ${NEW_PREFIX}_best.ckpt \
    --output outputs/gsparam_finetune/new/eval.mp4

echo "=== OLD baseline (gs_mask_model_best.ckpt) — for reference ==="
python -u eval_segmentation_gs_mask.py --npz ${CLIPS} \
    --gs_mask_head_path models/NeoVerse/gs_mask_model_best.ckpt \
    --output outputs/gsparam_finetune/old/eval.mp4

echo "FINETUNE+EVAL DONE."
