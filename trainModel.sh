#!/bin/bash

#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 24:00:00
#SBATCH --mem=200G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere


source env-gpu/bin/activate

python3 -u trainModelSparse.py
