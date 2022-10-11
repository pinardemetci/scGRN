#!/bin/bash

#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 24:00:00
#SBATCH --mem=130G
#SBATCH --job-name tunnel
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --output jupyter-log-%J.txt
## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)
## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport $USER@ssh.ccv.brown.edu
    -----------------------------------------------------------------
    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "
## start an ipcluster instance and launch jupyter server
jupyter notebook --no-browser --port=$ipnport --ip=$ipnip
