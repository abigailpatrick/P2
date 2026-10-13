#!/bin/bash
#SBATCH --job-name=extract_subcubes
#SBATCH --partition=all
#SBATCH --output=extract_subcubes_%j.log
#SBATCH --error=extract_subcubes_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

# Cuts MUSE_subcubes/<ID>/<ID>_subcube.fits for every source in
# jwst_catalogs/primer_minerva_in_muse_zsys.csv. One job, sources in series,
# since every source reads the same 200 GB cube and the run writes one log CSV.
#
# Submit from /ceph/cephfs/apatrick/P2/scripts/slurm so the .log and .err land there:
#   sbatch run_extract_subcubes.slurm

. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39

cd /ceph/cephfs/apatrick/P2/scripts/

echo "======================================="
echo "extract_subcubes started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "======================================="

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/usr/bin/time -v python -u extract_subcubes.py

echo "======================================="
echo "Job finished at: $(date)"
echo "======================================="