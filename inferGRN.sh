#!/bin/bash

#SBATCH -n 1
#SBATCH -c 32
#SBATCH --time 48:00:00
#SBATCH --mem=220G
python3 inferGRN.py
