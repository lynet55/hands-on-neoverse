#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=benchmark_seg_compare
#SBATCH --partition=jobs
#SBATCH --time=7-00:00:00      # 7 days (max allowed)
#SBATCH --chdir=/work/courses/3dv/team32/hands-on-neoverse
#SBATCH --output=/work/courses/3dv/team32/hands-on-neoverse/logs/benchmark_logs/%x_%j.out
#SBATCH --error=/work/courses/3dv/team32/hands-on-neoverse/logs/benchmark_logs/%x_%j.err

# ---------- Modules ----------
. /etc/profile.d/modules.sh
module add cuda/12.8

# ---------- Reporting ----------
echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node:          $SLURMD_NODENAME"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "GPUs:          ${SLURM_GPUS_ON_NODE:-?} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?})"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-default}"
echo "Memory:        ${SLURM_MEM_PER_NODE:-default} MB"
echo "Working dir:   $(pwd)"
echo "Start time:    $(date)"
echo "------------------------------------------"
nvcc --version
nvidia-smi || true
echo "=========================================="

# Fail fast on errors; show piped command failures
set -eo pipefail

# ---------- Environment ----------
source ./neoverse/bin/activate
echo "Python:        $(which python) ($(python --version 2>&1))"
echo "------------------------------------------"

mkdir -p logs/benchmark_logs runs/benchmarks

# ---------- Run ----------
# seg_compare: hand_seg (hand_pred_head -> seg_labels) vs gs_mask (gs_head -> rasterized
# mask channels), both on the shared base reconstructor, scored with the same seg metrics.
START=$(date +%s)

python -u -m diffsynth.data.benchmarking.benchmark_rec \
    --mode seg_compare \
    --reconstruction-model-path models/NeoVerse/reconstructor.ckpt \
    --hand-head-path models/NeoVerse/hand_seg_model_opt_best.ckpt \
    --gs-mask-head-path models/NeoVerse/gs_mask_model_run20260510-175056_epoch006.ckpt \
    --label-a hand_seg \
    --label-b gs_mask \
    --data-root /work/courses/3dv/team32/training_data_modal \
    --val-fraction 0.1 \
    --frame-stride 3 \
    --window-size 6 \
    --img-shape 280 280 \
    --num-classes 4 \
    --num-workers 2 \
    --output-dir runs/benchmarks
EXIT_CODE=$?

END=$(date +%s)
echo "------------------------------------------"
echo "Exit code:     $EXIT_CODE"
echo "Duration:      $((END - START)) seconds"
echo "End time:      $(date)"
echo "=========================================="

exit $EXIT_CODE
