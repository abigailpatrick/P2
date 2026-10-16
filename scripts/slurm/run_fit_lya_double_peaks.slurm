#!/bin/bash
#SBATCH --job-name=lya_double
#SBATCH --partition=all
#SBATCH --output=lya_double_%j.log
#SBATCH --error=lya_double_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39
cd /ceph/cephfs/apatrick/P2/scripts/
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "fit_lya_double_peak (job ${SLURM_JOB_ID}) on ${SLURMD_NODENAME} started $(date)"
python -u fit_lya_double_peak.py --n-mc 500
echo "Finished at: $(date)"