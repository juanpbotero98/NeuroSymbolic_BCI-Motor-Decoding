#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=notchpeak-gpu
#SBATCH --account=notchpeak-gpu
#SBATCH --gres=gpu:3090:1
#SBATCH --time=1-00:00:00

# load your module
module --force purge
module load miniforge3/latest

# activate conda environment
conda init 
conda activate Neurosymbolic
# run training script
python evaluate_GRU_RNN_Classifier.py --gpu True
