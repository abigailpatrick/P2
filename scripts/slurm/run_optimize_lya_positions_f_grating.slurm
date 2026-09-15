#!/bin/bash
#SBATCH --job-name=lya_falsepos_opt
#SBATCH --partition=all
#SBATCH --output=falsepos_opt_%j.log
#SBATCH --error=falsepos_opt_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Load and activate conda
. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39

cd /ceph/cephfs/apatrick/P2/scripts/

echo "======================================="
echo "Lya false-positive optimiser started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "======================================="

# Disable OpenMP parallelism in math libs to avoid thread contention
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

FALSE_DIR=/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts/false_positions
CUBE_DIR=/ceph/cephfs/apatrick/P2/MUSE_subcubes/contsub/
KEPT_CSV=${FALSE_DIR}/false_positions_kept.csv

# Execute the optimiser over all kept sources
/usr/bin/time -v python optimize_lya_positions_f_grating.py \
  --all \
  --false-dir ${FALSE_DIR} \
  --cube-dir ${CUBE_DIR} \
  --outdir ${FALSE_DIR} \
  --kept-csv ${KEPT_CSV} \
  --aperture 0.6 \
  --dx-max 0.4 \
  --dx-step 0.1 \
  --prior-scale 1.0 \
  2>&1 | tee falsepos_opt_${SLURM_JOB_ID}_runtime.log

echo "======================================="
echo "Job finished at: $(date)"
echo "======================================="