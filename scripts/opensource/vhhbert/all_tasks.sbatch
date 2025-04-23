#!/bin/bash
#SBATCH --job-name=vhhbert_tasks
#SBATCH --output=logs/all_tasks_%j.out
#SBATCH --error=logs/all_tasks_%j.err
#SBATCH --partition=peacock05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00

# Debug information
echo "Starting job at $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"

# Load required modules
echo "Loading modules..."
module list
module load cuda/11.8
module load python/3.8
module list

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate your conda environment if needed
# conda activate your_env_name

# Change to the script directory
cd /home/yzhang/research/nanobody/scripts/opensource/vhhbert

# Check if the script exists and is executable
if [ ! -f "./all_tasks.sh" ]; then
    echo "Error: all_tasks.sh not found!"
    exit 1
fi

if [ ! -x "./all_tasks.sh" ]; then
    echo "Error: all_tasks.sh is not executable!"
    exit 1
fi

# Execute the script
echo "Executing all_tasks.sh..."
./all_tasks.sh

echo "Job completed at $(date)" 