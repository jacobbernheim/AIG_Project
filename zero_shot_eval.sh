#!/bin/bash
#SBATCH --partition gpu4_short,gpu4_dev,gpu8_short,gpu8_dev,a100_long,a100_short
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 20G
#SBATCH --gres=gpu:1
#SBATCH --time 0-04:00:00
#SBATCH --job-name eval-ag
#SBATCH --output logs/eval-ag-%J.log 

source /gpfs/data/shenhavlab/users/ca3261/set_r_conda.sh

conda activate sox2-alphagenome
conda info
echo "conda activated"

python zero_shot_eval.py \
    --zero-shot-csv /gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/data/zero_shot/zero_shot_scores.csv \
    --categories "Sub-DHS Level" "DHS Level" \
    --score-col normalized_score \
    --output-dir /gpfs/scratch/ca3261/ai_in_genomics/final_project/AIG_Project/results/eval_zero_shot/