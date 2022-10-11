#!/bin/bash

#SBATCH -p 3090-gcondo,gpu --gres=gpu:1

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 2 CPU cores
#SBATCH -n 1
#SBATCH --mem=40g
#SBATCH --time=10:00:00

source env/bin/activate
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html ; pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
