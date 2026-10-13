#!/bin/bash
#SBATCH --job-name=lya_mc_merge
#SBATCH --partition=all
#SBATCH --output=mc_merge_%j.log
#SBATCH --error=mc_merge_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Run after the MC array, with --dependency=afterok:<array job id>.
# Joins MUSE_subcubes/<ID>/<ID>_lya_mc.csv onto MUSE_catalogs/lya_fits.csv
# and writes MUSE_catalogs/lya_fits_mc.csv.

. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39

cd /ceph/cephfs/apatrick/P2/scripts/

echo "Merging MC results at: $(date)"
python -u merge_mc_partials.py
echo "Merge finished at: $(date)"