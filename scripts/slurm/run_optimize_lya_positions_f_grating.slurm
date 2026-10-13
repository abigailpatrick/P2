#!/bin/bash
#SBATCH --job-name=lya_falsepos
#SBATCH --partition=all
#SBATCH --output=falsepos_%A_%a.log
#SBATCH --error=falsepos_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-90

# One task per source of jwst_catalogs/primer_minerva_in_muse_zsys.csv, sorted
# by ID. Task N runs the false positions of the Nth ID and writes
# MUSE_subcubes/<ID>/<ID>_false_optimal_snr.csv. Sources left out by
# false_pos_grating.py (e.g. no band coverage) exit at once.
# Run false_pos_grating.py first. Submit from scripts/slurm with
#   sbatch --array=0-90 run_optimize_lya_positions_f_grating.slurm
# then, once every task has finished, run false_pos_percentiles.py.

. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39

cd /ceph/cephfs/apatrick/P2/scripts/

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "======================================="
echo "False positions task ${SLURM_ARRAY_TASK_ID} (job ${SLURM_ARRAY_JOB_ID})"
echo "Node: ${SLURMD_NODENAME}  started: $(date)"
echo "======================================="

python -u optimize_lya_positions_f_grating.py --task-index ${SLURM_ARRAY_TASK_ID}

echo "Task ${SLURM_ARRAY_TASK_ID} finished at: $(date)"