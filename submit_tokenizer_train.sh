#!/bin/bash
# This is the outer script that submits the inner script to Slurm.
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=tokenizer_train       # Job name
#SBATCH --time=0-02:00:00                # Max run time (e.g., 2 hours, adjust as needed)
#SBATCH --nodes=1                        # Request 1 node
#SBATCH --ntasks=1                       # Request 1 task (process)
#SBATCH --cpus-per-task=10               # Request 10 CPU cores (adjust if tokenizer needs less)
#SBATCH --mem=32000                      # Request 32GB of RAM (adjust if tokenizer needs less)
#SBATCH --output=tokenizer_train_%j.out  # Standard output and error log
#SBATCH --error=tokenizer_train_%j.err   # Standard error log
#SBATCH --account=lz25                   # Your project account
#SBATCH --export=NONE                    # Prevents unnecessary environment variables from propagating
#SBATCH --partition=m3g                  # Your GPU partition
#SBATCH --gres=gpu:V100:1                # Request 1 V100 GPU (for tokenizers library speedup)
#SBATCH --mail-user=ext-ntyagi@monash.edu # Your email for notifications
#SBATCH --mail-type=ALL                  # Get email for ALL events (BEGIN, END, FAIL)

#======START=====
date
echo "Starting Tokenizer Training Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"

# --- Environment Setup ---
source /projects/lz25/navyat/conda
conda activate /projects/lz25/navyat/conda/envs/genomeenv # <--- ADJUST THIS PATH/ENV NAME IF DIFFERENT

echo "Conda environment activated: $(conda env list | grep '*' | awk '{print $NF}')"
echo "Python interpreter: $(which python)"
nvidia-smi # Display GPU status

# --- Run the Python Tokenizer Training Script ---
echo "Executing Python tokenizer training script..."
python auto_continual_tokenise_bio.py

# --- End Job ---
echo "Python script finished."
date
exit 0
EOT
