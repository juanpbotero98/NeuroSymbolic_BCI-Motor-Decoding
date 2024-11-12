#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --mem=0
#SBATCH --partition=notchpeak-shared-short
#SBATCH --account=notchpeak-gpu
#SBATCH --gres=gpu:m2090:2
#SBATCH --time=0:05:00
