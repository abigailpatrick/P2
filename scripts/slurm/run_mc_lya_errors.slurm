#!/bin/bash
#SBATCH --job-name=lya_mc_err
#SBATCH --partition=all
#SBATCH --output=mc_err_%A_%a.log
#SBATCH --error=mc_err_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-81

# One array task per successfully-fitted source. Set --array=0-(N-1) where N is
# the number of fit_success rows in lya_properties.csv. If unsure, run
#   python -c "import pandas as pd; d=pd.read_csv('<properties_csv>'); \
#     print(d['fit_success'].astype(str).str.lower().isin(['true','1']).sum())"
# and set the upper bound to that count minus one. Extra tasks past the end
# simply exit as no-ops, so a slightly-too-large range is harmless.

. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39

cd /ceph/cephfs/apatrick/P2/scripts/

# Keep each task single-threaded so 82 of them don't oversubscribe the node
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

INDIR=/ceph/cephfs/apatrick/P2/MUSE_subcubes/dataproducts
SNR_CSV=/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_sliding_snr_grating.csv
CATALOG=/ceph/cephfs/apatrick/P2/jwst_catalogs/grating_sources_with_zsys.csv
PROPS=/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties.csv
PARTIALS=/ceph/cephfs/apatrick/P2/MUSE_catalogs/mc_partials
OUT=${PARTIALS}/mc_row${SLURM_ARRAY_TASK_ID}.csv

mkdir -p ${PARTIALS}

echo "======================================="
echo "MC error task ${SLURM_ARRAY_TASK_ID} (job ${SLURM_ARRAY_JOB_ID})"
echo "Node: ${SLURMD_NODENAME}  started: $(date)"
echo "======================================="

python mc_lya_errors_grating.py \
  --indir ${INDIR} \
  --snr-csv ${SNR_CSV} \
  --catalog ${CATALOG} \
  --properties-csv ${PROPS} \
  --out-csv ${OUT} \
  --n-mc 500 --mode model \
  --row-index ${SLURM_ARRAY_TASK_ID}

echo "Task ${SLURM_ARRAY_TASK_ID} finished at: $(date)"