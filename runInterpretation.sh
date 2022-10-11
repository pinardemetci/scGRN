#!/bin/bash

#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time 30:00:00
#SBATCH --mem=70G

python3 -u runInterpretationNew.py

