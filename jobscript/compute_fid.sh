#!/bin/bash -l
#$ -N compute_fid
#$ -l h_rt=2:0:0
#$ -cwd
#$ -l mem=100G
#$ -l gpu=1
#$ -ac allow=L
#$ -m ea
#$ -M xinyuan.chen.24@ucl.ac.uk

unset PYTHONPATH
module purge

module load python/miniconda3/24.3.0-0
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate galaxy

# shellcheck disable=SC2164
cd /home/ucaphey/Scratch/galaxy_ddpm
python -m metrics.fid