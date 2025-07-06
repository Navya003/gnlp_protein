#!/bin/bash
# This is the outer script that submits the inner script to Slurm.
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=modernbert_pretrain      # Job name for squeue, etc.
#SBATCH --time=1-00:00:00                   # Max run time (1 day, matches your previous)
#SBATCH --nodes=1                           # Request 1 node
#SBATCH --ntasks=1                          # Request 1 task (process)
#SBATCH --cpus-per-task=20                  # Request 20 CPU cores (matches your previous)
#SBATCH --mem=64000                         # Request 64GB of RAM (matches your previous)
#SBATCH --output=modernbert_pretrain_%j.out # Standard output and error log
#SBATCH --error=modernbert_pretrain_%j.err  # Standard error log
#SBATCH --account=lz25                      # Your project account (matches your previous)
#SBATCH --export=NONE                       # Prevents unnecessary environment variables from propagating
#SBATCH --partition=m3g                     # Your GPU partition (matches your previous)
#SBATCH --gres=gpu:V100:1                   # Request 1 V100 GPU (matches your previous)
##SBATCH --partition=genomics              # Uncomment if you want to use the genomics partition
##SBATCH --qos=genomics                    # Uncomment if you want to use the genomics QOS
#SBATCH --mail-user=ext-ntyagi@monash.edu   # Your email for notifications
#SBATCH --mail-type=ALL                     # Get email for ALL events (BEGIN, END, FAIL)

#======START=====
date
echo "Starting ModernBERT Pre-training Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"

# --- Environment Setup ---
# Source your base conda environment
source /projects/lz25/navyat/conda

# Activate the *specific* conda environment for ModernBERT pre-training
# IMPORTANT: Replace 'gnlp' with the name of the conda environment
# where you installed all the dependencies for the ModernBERT script
# (e.g., 'modernbert_pretrain_env' or a new one you create for this project).
conda activate /projects/lz25/navyat/conda/envs/modernbert_py310

# Check activated environment (optional, for debugging)
echo "Conda environment activated: $(conda env list | grep '*' | awk '{print $NF}')"
echo "Python interpreter: $(which python)"
nvidia-smi # Display GPU status

# --- Run the Python Pre-training Script ---
# Make sure 'pretrain_modernbert.py' is in the current directory or provide its full path.
# And ensure paths inside 'pretrain_modernbert.py' are correct for the HPC environment.
echo "Executing Python pre-training script..."
python pretrain_modernbert.py # Execute the python script directly

# OR, if you made it executable using chmod +x:
# ./pretrain_modernbert.py

# --- End Job ---
echo "Python script finished."
date
exit 0
EOT
