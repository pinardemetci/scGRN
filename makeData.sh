#!/bin/sh
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --time 48:00:00
#SBATCH --mem=30G

python3 dataset.py
