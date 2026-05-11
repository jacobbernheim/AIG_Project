#!/bin/bash
#SBATCH --partition gpu4_short,gpu4_dev,gpu8_short,gpu8_dev,a100_long,a100_short
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 20G
#SBATCH --gres=gpu:1
#SBATCH --time 0-04:00:00
#SBATCH --job-name finetune-ag
#SBATCH --output logs/finetune-ag-%J.log 

source /gpfs/data/shenhavlab/users/ca3261/set_r_conda.sh

conda activate sox2-alphagenome
conda info
echo "conda activated"

python finetune.py \
    --data-file "/gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/processed/merged_payloads.csv" \
    --output-dir /gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/dhs_finetune_updated/ \
    --category "DHS Level"
# Sub-DHS Level, DHS Level
