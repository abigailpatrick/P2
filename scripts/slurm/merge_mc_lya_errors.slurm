#!/bin/bash
#SBATCH --job-name=lya_mc_merge
#SBATCH --partition=all
#SBATCH --output=mc_merge_%j.log
#SBATCH --error=mc_merge_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Run after the MC array finishes. Submit with a dependency, e.g.:
#   AID=$(sbatch --parsable run_mc_lya_errors.slurm)
#   sbatch --dependency=afterok:${AID} merge_mc_lya_errors.slurm
# so this only starts once every array task has completed successfully.

. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39

cd /ceph/cephfs/apatrick/P2/scripts/

PARTIALS=/ceph/cephfs/apatrick/P2/MUSE_catalogs/mc_partials
PROPS=/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties.csv
OUT=/ceph/cephfs/apatrick/P2/MUSE_catalogs/lya_properties_mc.csv

echo "Merging MC partials at: $(date)"

python merge_mc_partials.py \
  --partials-dir ${PARTIALS} \
  --properties-csv ${PROPS} \
  --out-csv ${OUT}

echo "Merge finished at: $(date)"