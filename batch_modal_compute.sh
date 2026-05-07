#!/bin/bash
#SBATCH --account=3dv
#SBATCH --partition=interactive-cpu
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/modal_compute_%j.out

# Downloads and processes clips 400-end into training_data_modal/.
# Submit with: sbatch batch_modal_compute.sh

set -eo pipefail

cd /work/courses/3dv/team32/handy-NeoVerse
source ./neoverse/bin/activate

mkdir -p logs

python diffsynth/data/optimized_download.py \
  --clip-start 400 \
  --tar-dir diffsynth/data/tar_recv_modal_1 \
  --output-dir diffsynth/data/training_data_modal \
  --mask-type modal \
  --num-workers 2 --max-stored-clips 4
