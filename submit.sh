#! /bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=medium
#SBATCH --clusters=all

#SBATCH --gres=gpu:2 --constraint='gpu_mem:48GB'

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=andrew.bean@oii.ox.ac.uk

module load Anaconda3/2020.11
source activate $DATA/inference

python baseline_script.py
