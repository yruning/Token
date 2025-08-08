#!/bin/bash
#SBATCH -A llmalignment         # Account name
#SBATCH -t 5:00:00             # Time limit
#SBATCH -p a100_normal_q        # Partition
#SBATCH -N 1                    # Number of nodes
#SBATCH --gres=gpu:1            # GPU resource
#SBATCH --mem=40G               # Memory

# Load your modules and activate environment if needed
#module load anaconda
source activate myenv  # replace with your conda env

# Run your Python script
python tag_collect.py --data_name HH --model_name phi-2-tag --strategy basic --model_para phi-2
