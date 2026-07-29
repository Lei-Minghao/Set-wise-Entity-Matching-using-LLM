#!/bin/bash
#SBATCH --job-name=llm_ranking
#SBATCH --output=llm_ranking_%j.out
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=12         
#SBATCH --mem=70G
#SBATCH --time=02:00:00

# 1. Environment & Module Setup
source   # <-- CHANGE THIS TO YOUR SOURCE
conda activate # <-- CHANGE THIS TO YOUR ENVIRONMENT

# Define the absolute path to your environment to bypass fickle conda variables
ENV_LIB=""
export CUDA_HOME=/usr/local/cuda

# Force the loader to search your conda environment's folder FIRST
export LD_LIBRARY_PATH=

# 2. Run the script directly
echo "Starting direct inference job..."
"python environment" ranking_qwen_WDC.py

echo "Job finished."