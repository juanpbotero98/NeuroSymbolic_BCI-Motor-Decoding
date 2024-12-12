#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=notchpeak-gpu
#SBATCH --account=notchpeak-gpu
#SBATCH --gres=gpu:3090:1
#SBATCH --time=0-05:00:00

# load your module
module --force purge
module load miniforge3/latest

# activate conda environment
conda init 
conda activate Neurosymbolic

# Evaluate the GRU RNN Classifier
python evaluate_GRU_RNN_Classifier.py --gpu true 

# Evaluate the GRU RNN Regressor Pos
# python evaluate_GRU_RNN_Regressor.py --gpu true --model_name 'gru_regressor_pos-ls128-sql1000-15epochs' --decoded_var 'pos'

# Evaluate the GRU RNN Regressor Vel
python evaluate_GRU_RNN_Regressor.py --gpu true --model_name 'gru_regressor_vel-ls128-sql1000-15epochs' --decoded_var 'vel'

