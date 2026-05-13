#!/bin/bash
#SBATCH --partition gpu4_dev
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 20G
#SBATCH --gres=gpu:1
#SBATCH --time 0-00:10:00
#SBATCH --job-name finetune-ag
#SBATCH --output logs/finetune-ag-%J.log 

source /gpfs/data/shenhavlab/users/ca3261/set_r_conda.sh

conda activate sox2-alphagenome
conda info
echo "conda activated"

python finetune.py \
    --data-file "/gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/processed/merged_payloads.csv" \
    --output-dir /gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/results/dhs_finetune/ \
    --category "DHS Level" \
    --split-seed 22

python finetune.py \
    --data-file "/gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/processed/merged_payloads.csv" \
    --output-dir /gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/results/sub_dhs_finetune/ \
    --category "Sub-DHS Level" \
    --split-seed 22

## 555, 96, 6721, 888 (good mlp but bad cv)
python final_eval.py
