#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=gs_mask_smoke
#SBATCH --partition=jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=0-00:30:00
#SBATCH --chdir=/work/courses/3dv/team32/hands-on-neoverse
#SBATCH --output=/work/courses/3dv/team32/hands-on-neoverse/logs/gs_mask_smoke_%j.out
#SBATCH --error=/work/courses/3dv/team32/hands-on-neoverse/logs/gs_mask_smoke_%j.err

mkdir -p logs
. /etc/profile.d/modules.sh
module add cuda/12.8
set -eo pipefail
source ./neoverse/bin/activate
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi || true

python -u smoke_gs_mask_toggles.py
echo "SMOKE JOB DONE."
