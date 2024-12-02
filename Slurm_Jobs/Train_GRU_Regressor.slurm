#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=notchpeak-gpu
#SBATCH --account=notchpeak-gpu
#SBATCH --gres=gpu:3090:1
#SBATCH --time=0-12:00:00

# load your module
module --force purge
module load miniforge3/latest

# activate conda environment
conda init 
conda activate Neurosymbolic

# run training script
python train_GRU_RNN.py --epochs 15 --objective 'pos' --model_path 'Training_logs/20241130-154128-GRU_regressor_pos-ls128-lr0.001-bs32-sl1000-15epochs/gru_rnn_checkpoint.pth' --log_dir 'Training_logs/20241130-154128-GRU_regressor_pos-ls128-lr0.001-bs32-sl1000-15epochs' 
