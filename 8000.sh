#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=8000.sweep
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64000
#SBATCH --output=8000.sweep.out
#SBATCH --account=lz25
#SBATCH --export=NONE
#SBATCH --partition=m3g
#SBATCH --gres=gpu:V100:1
##SBATCH --partition=genomics
##SBATCH --qos=genomics
#SBATCH --mail-user=ext-ntyagi@monash.edu
#SBATCH --mail-type=ALL
#======START=====
source /projects/lz25/navyat/conda
conda activate /projects/lz25/navyat/conda/envs/something

date


MODEL="distilbert"
SCOUNT=128

sweep \
  prot/8000_data.csv/train.parquet \
  "parquet" \
  prot/8000_tokens.json \
  --test prot/8000_data.csv/test.parquet \
  --valid prot/8000_data.csv/valid.parquet \
  --hyperparameter_sweep params.json \
  --entity_name tyagilab \
  --project_name p_sweep \
  --group_name vocab_8000 \
  --output_dir p_sweep_8000 \
  --label_names "labels" \
  --model distilbert \
  --sweep_count 128 \
  --metric_opt "eval/f1"

date
exit 0
EOT
