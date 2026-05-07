#!/bin/bash
# Run inside a tmux session on the login node.
# Downloads and processes clips 0-399 into training_data_modal/.
#
# Usage:
#   tmux new -s modal_login
#   bash batch_modal_login.sh

set -eo pipefail

cd /work/courses/3dv/team32/handy-NeoVerse
source ./neoverse/bin/activate

mkdir -p logs

python diffsynth/data/optimized_download.py \
  --clip-start 0 --clip-end 399 \
  --tar-dir diffsynth/data/tar_recv_modal_0 \
  --output-dir diffsynth/data/training_data_modal \
  --mask-type modal \
  --num-workers 2 --max-stored-clips 4
