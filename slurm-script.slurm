#!/bin/bash  
#SBATCH --job-name=ntd_train                # A name for your job 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=8                    # Number of CPUs  
#SBATCH --mem=32G                            # Memory: adjust based on your data/model size  
#SBATCH --time=24:00:00                      # Wall‑time limit (e.g., 24 hours)  
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --mail-type=END,FAIL                 # Get email notifications of the state of the job.
#SBATCH --mail-user=da330180@ucf.edu
#SBATCH --output=ntd_train_%j.out            # Standard output log  
#SBATCH --error=ntd_train_%j.err             # Standard error log

echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# Load Modules
#module purge
module load anaconda/anaconda-2023.09

# Navigate to project directory
cd ~/ntd/neural_timeseries_diffusion || exit 1

# activate Environment
conda activate ./ntd_env

# Set PYTHONPATH to include the repo root
export PYTHONPATH=$PYTHONPATH:.

# generate the index and subject jsons
python3 prepare_data.py

echo "Starting training..."

python3 -m ntd.train_diffusion_model \
    dataset=cogpilot \
    network=ada_conv_cogpilot \
    diffusion=diffusion_linear_500 \
    optimizer=base_optimizer \
    dataset.index_json="data/processed/dataset_index.json" \
    dataset.signal_length=512 \
    optimizer.num_epochs=250 \
    optimizer.train_batch_size=32 \
    base.experiment="debug_run" \
    base.use_cuda_if_available=True \
    +generate_samples=null
echo "Training Complete."

python3 generate_cogpilot.py \
    dataset=cogpilot \
    network=ada_conv_cogpilot \
    diffusion=diffusion_linear_500 \
    base.use_cuda_if_available=True \
    base.experiment="debug_run"

echo "Generation Complete."
