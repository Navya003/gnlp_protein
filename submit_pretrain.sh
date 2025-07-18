#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=modernbert_pretrain
#SBATCH --time=2-00:00:00         # Increased time limit to 2 days
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64000
#SBATCH --output=modernbert_pretrain_%j.out
#SBATCH --error=modernbert_pretrain_%j.err
#SBATCH --account=lz25
#SBATCH --export=NONE             # Keeping this as it is
#SBATCH --partition=m3g
#SBATCH --gres=gpu:V100:1
##SBATCH --partition=genomics
##SBATCH --qos=genomics
#SBATCH --mail-user=ext-ntyagi@monash.edu
#SBATCH --mail-type=ALL

#======START=====
date
echo "Starting ModernBERT Pre-training Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Running on host: $(hostname)"

# --- Navigate to working directory ---
cd /home/navyat/nt # Added command to navigate to the working directory
echo "Current directory: $(pwd)"

# --- Environment Setup ---
source /projects/lz25/navyat/conda/etc/profile.d/conda.sh
conda activate /projects/lz25/navyat/conda/envs/modernbert_py310

echo "Conda environment activated: $(conda env list | grep '*' | awk '{print $NF}')"
echo "Python interpreter: $(which python)"

# Load CUDA module for GPU utilization
module load cuda # Added to ensure GPU use

nvidia-smi # Display GPU status

# --- Run the Python Pre-training Script ---
echo "Executing Python pre-training script..."
python pretrain_modernbert.py

# --- End Job ---
echo "Python script finished."
date
exit 0
EOT
