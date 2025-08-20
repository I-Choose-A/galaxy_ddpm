#!/bin/bash -l
#$ -N train_inception_v3
#$ -l h_rt=48:0:0
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
python -m FID.train_inception