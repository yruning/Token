#!/bin/bash
#SBATCH -A llmalignment         # Account name
#SBATCH -t 24:00:00             # Time limit
#SBATCH -p a100_normal_q         # Partition
#SBATCH -N 1                    # Number of nodes
#SBATCH --gres=gpu:1            # GPU resource
#SBATCH --mem=80G               # Memory

# Load your modules and activate environment if needed
#module load anaconda
module load CUDA/12.1.1
source activate llmrl  # replace with your conda env

# Run your Python script
python DPO_tag.py --data_name HH --model_name phi-2 --r_method dpo_tag
