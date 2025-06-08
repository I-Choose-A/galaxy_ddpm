#!/bin/bash -l

#$ -N train_unconditional_ddpm
#$ -l h_rt=5:0:0
#$ -cwd
#$ -l mem=100G
#$ -l gpu=1
#$ -m ea
#$ -M xinyuan.chen.24@ucl.ac.uk

module unload gcc-libs

module load python3/3.9-gnu-10.2.0

module load pytorch/2.1.0/gpu

python main.py