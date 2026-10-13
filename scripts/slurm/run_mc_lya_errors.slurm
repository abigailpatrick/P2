#!/bin/bash
#SBATCH --job-name=lya_mc_err
#SBATCH --partition=all
#SBATCH --output=mc_err_%A_%a.log
#SBATCH --error=mc_err_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-90

# One task per source of jwst_catalogs/primer_minerva_in_muse_zsys.csv, sorted
# by ID. Task N does the Nth ID and writes MUSE_subcubes/<ID>/<ID>_lya_mc.csv.
# Sources without a successful fit exit at once. Set the array to the size of
# the sample when submitting, e.g. from scripts/slurm
#   N=$(python -c "import pandas as pd; print(len(pd.read_csv('/ceph/cephfs/apatrick/P2/jwst_catalogs/primer_minerva_in_muse_zsys.csv')))")
#   AID=$(sbatch --parsable --array=0-$((N-1)) run_mc_lya_errors.slurm)
#   sbatch --dependency=afterok:${AID} merge_mc_lya_errors.slurm

. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39

cd /ceph/cephfs/apatrick/P2/scripts/

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "======================================="
echo "MC error task ${SLURM_ARRAY_TASK_ID} (job ${SLURM_ARRAY_JOB_ID})"
echo "Node: ${SLURMD_NODENAME}  started: $(date)"
echo "======================================="

python -u mc_lya_errors_grating.py --n-mc 500 --mode model --task-index ${SLURM_ARRAY_TASK_ID}

echo "Task ${SLURM_ARRAY_TASK_ID} finished at: $(date)"