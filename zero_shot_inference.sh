#!/bin/bash
#SBATCH --partition gpu4_short,gpu4_dev,gpu8_short,gpu8_dev,a100_long,a100_short
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 20G
#SBATCH --gres=gpu:1
#SBATCH --time 0-04:00:00
#SBATCH --job-name zero-shot-ag
#SBATCH --output logs/zero-shot-ag-%J.log 

source /gpfs/data/shenhavlab/users/ca3261/set_r_conda.sh

conda activate sox2-alphagenome
conda info
echo "conda activated"

# python /gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/example_inference.py
# in this context can we not just use payload activities because it has the PL and name cols we need?
python zero_shot_inference.py \
    --data-file "/gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/processed/merged_payloads.csv" \
    --output-dir /gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/zero_shot/
