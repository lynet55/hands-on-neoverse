#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=gs_mask_unfreeze
#SBATCH --partition=jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=0-05:00:00
#SBATCH --chdir=/work/courses/3dv/team32/hands-on-neoverse
#SBATCH --output=/work/courses/3dv/team32/hands-on-neoverse/logs/gs_mask_unfreeze_%j.out
#SBATCH --error=/work/courses/3dv/team32/hands-on-neoverse/logs/gs_mask_unfreeze_%j.err

# #4 capacity run: fine-tune from best with last-2 backbone blocks unfrozen,
# + Lovász(0.1) + geometry-detach, low LR, 5000 steps. Saves step checkpoints,
# then evaluates the whole trajectory on the 3 clips to pick the true best.
mkdir -p logs
. /etc/profile.d/modules.sh; module add cuda/12.8
set -eo pipefail; source ./neoverse/bin/activate
echo "Python: $(which python)"; nvidia-smi || true

PFX=models/NeoVerse/gs_mask_unfreeze
export GSMASK_DATA_ROOT=/work/courses/3dv/team32/training_data_modal
export GSMASK_RESUME_FROM=models/NeoVerse/gs_mask_model_best.ckpt
export GSMASK_FINETUNE=1
export GSMASK_RUN_ID=unfreeze2
export GSMASK_SAVE_PREFIX=${PFX}
export GSMASK_LOVASZ=0.1
export GSMASK_DETACH_GEOM=1
export GSMASK_UNFREEZE_N=2
export GSMASK_LR=1e-5
export GSMASK_EPOCHS=50
export GSMASK_MAX_STEPS=5000
export GSMASK_SAVE_EVERY=1000

python -u -m diffsynth.data.training.training_gs_mask

CLIPS="/work/courses/3dv/team32/training_data_modal/clip-001100.npz /work/courses/3dv/team32/training_data_modal/clip-001270.npz /work/courses/3dv/team32/training_data_modal/clip-001284.npz"
echo ""; echo "########## RE-EVAL unfreeze trajectory vs baseline (0.768/0.617/0.618) ##########"
for CK in ${PFX}_step1000.ckpt ${PFX}_step2000.ckpt ${PFX}_step3000.ckpt ${PFX}_step4000.ckpt ${PFX}_step5000.ckpt ${PFX}_latest.ckpt; do
  [ -f "$CK" ] || continue
  echo "=== CKPT: $CK ==="
  python -u eval_segmentation_gs_mask.py --npz ${CLIPS} --gs_mask_head_path "$CK" --output outputs/unfreeze_eval/$(basename $CK .ckpt)/eval.mp4
done
echo "UNFREEZE RUN DONE."
