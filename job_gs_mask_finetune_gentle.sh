#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=gs_mask_ft_gentle
#SBATCH --partition=jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=0-04:00:00
#SBATCH --chdir=/work/courses/3dv/team32/hands-on-neoverse
#SBATCH --output=/work/courses/3dv/team32/hands-on-neoverse/logs/gs_mask_ft_gentle_%j.out
#SBATCH --error=/work/courses/3dv/team32/hands-on-neoverse/logs/gs_mask_ft_gentle_%j.err

# Gentle re-probe: refine (not perturb) the best checkpoint. Lower LR (1e-5) and
# lower Lovász weight (0.1) than the first probe, keep #6 detach, slightly longer.
mkdir -p logs
. /etc/profile.d/modules.sh; module add cuda/12.8
set -eo pipefail; source ./neoverse/bin/activate
echo "Python: $(which python)"; nvidia-smi || true

NEW_PREFIX=models/NeoVerse/gs_mask_gentle
export GSMASK_DATA_ROOT=/work/courses/3dv/team32/training_data_modal
export GSMASK_RESUME_FROM=models/NeoVerse/gs_mask_model_best.ckpt
export GSMASK_FINETUNE=1
export GSMASK_RUN_ID=lovasz_detach_gentle
export GSMASK_SAVE_PREFIX=${NEW_PREFIX}
export GSMASK_LOVASZ=0.1
export GSMASK_DETACH_GEOM=1
export GSMASK_LR=1e-5
export GSMASK_EPOCHS=50
export GSMASK_MAX_STEPS=2000
export GSMASK_SAVE_EVERY=500

python -u -m diffsynth.data.training.training_gs_mask

CLIPS="/work/courses/3dv/team32/training_data_modal/clip-001100.npz /work/courses/3dv/team32/training_data_modal/clip-001270.npz /work/courses/3dv/team32/training_data_modal/clip-001284.npz"
echo ""; echo "########## RE-EVAL gentle: _latest (step 2000) vs _best vs baseline ##########"
echo "=== gentle _latest ==="
python -u eval_segmentation_gs_mask.py --npz ${CLIPS} --gs_mask_head_path ${NEW_PREFIX}_latest.ckpt --output outputs/gentle/latest/eval.mp4
echo "=== gentle _best (train-loss selected) ==="
python -u eval_segmentation_gs_mask.py --npz ${CLIPS} --gs_mask_head_path ${NEW_PREFIX}_best.ckpt --output outputs/gentle/best/eval.mp4
echo "GENTLE FT+EVAL DONE."
