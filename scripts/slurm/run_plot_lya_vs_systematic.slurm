#!/bin/bash
#SBATCH --job-name=lya_vs_sys
#SBATCH --partition=all
#SBATCH --output=lya_vs_sys_%j.log
#SBATCH --error=lya_vs_sys_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env312
cd /ceph/cephfs/apatrick/P2/scripts/
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "plot_lya_vs_systemic (job ${SLURM_JOB_ID}) on ${SLURMD_NODENAME} started $(date)"
python -u plot_lya_vs_systemic.py
echo "Finished at: $(date)"