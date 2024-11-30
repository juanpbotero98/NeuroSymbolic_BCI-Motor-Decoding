#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=notchpeak-gpu
#SBATCH --account=notchpeak-gpu
#SBATCH --gres=gpu:3090:1
#SBATCH --time=1-00:00:00

# load your module
module purge
module load miniforge3/latest

# activate conda environment
conda init 
conda activate Neurosymbolic

# run training script
python train_GRU_RNN.py --epochs 15 --objective 'pos'
