#!/bin/bash

# ==========================================
# Configuration Section
# ==========================================

# 1. Google Cloud / Vertex AI Configuration
export GOOGLE_CLOUD_PROJECT=""
export GOOGLE_CLOUD_LOCATION=""

# 2. Dataset Paths (relative to project root, adjust if needed)
export DATASET_PATH="./dataset/sample_walmart_amazon.csv"
export GROUND_TRUTH_PATH="./dataset/sample_walmart_amazon_gt.csv"

# 3. Model Configuration
# Path to local embedding model folder or Hugging Face name
export EMBEDDING_MODEL_PATH="all-MiniLM-L6-v2"
export GEMINI_MODEL="gemini-2.5-flash"

# ==========================================
# Execution Section
# ==========================================
echo "Starting Simplified LLMCER Pipeline..."
echo "Dataset: $DATASET_PATH"
echo "Ground Truth: $GROUND_TRUTH_PATH"
echo "Embedding Model: $EMBEDDING_MODEL_PATH"
echo "Gemini Model: $GEMINI_MODEL"
echo "GCP Project: $GOOGLE_CLOUD_PROJECT"
echo "GCP Location: $GOOGLE_CLOUD_LOCATION"

# Create logs directory if not exists
mkdir -p logs

# Extract dataset name for log file
DATASET_NAME=$(basename "$DATASET_PATH" | cut -d. -f1)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${DATASET_NAME}_simplified_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"

# Run the simplified pipeline (make sure run_pipeline.py is in the current directory or adjust path)
python -u llmcer_walmart_amazon.py | tee "$LOG_FILE"