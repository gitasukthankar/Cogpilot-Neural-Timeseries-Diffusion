#!/bin/bash  
#SBATCH --job-name=ntd_test                # A name for your job  
#SBATCH --cpus-per-task=8                    # Number of CPUs  
#SBATCH --mem=16G                            # Memory: adjust based on your data/model size  
#SBATCH --time=8:00:00                      # Wall‑time limit (e.g., 24 hours)  
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --mail-type=ALL                      # Get email notifications of the state of the job.
#SBATCH --mail-user=da330180@ucf.edu
#SBATCH --output=ntd_test%j.out            # Standard output log  
#SBATCH --error=ntd_test%j.err             # Standard error log


# Load Modules
#module purge
module load anaconda/anaconda-2023.09

cd ~/ntd/neural_timeseries_diffusion

# activate Environment
conda activate ./ntd_env

# Set PYTHONPATH to include the repo root
export PYTHONPATH=$PYTHONPATH:.

python3 -m test.test_generation