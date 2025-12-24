#!/bin/bash
#SBATCH -N 1 # Number of Nodes
#SBATCH --job-name=lmkt_train
#SBATCH --output=lmkt_%j.out
#SBATCH --error=lmkt_%j.err

#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100|a40|rtx8000|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 3-00:00:00
#SBATCH --qos=long
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules (adjust to your cluster)
module load conda/latest
module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6

# Activate your environment
conda activate LLMKT

# Optional: make sure the script stops on errors
set -e

# Run your command
python -m dialogue_kt.main train \
  --dataset eedi \
  --tag_src gpt \
  --crossval \
  --model_type lmkt \
  --model_name lmkt_eedi_gpt \
  --quantize 1
