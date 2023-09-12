#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=32000.sweep
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64000
#SBATCH --output=32000.sweep.out
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

vocab_size=32000
MODEL="distilbert"
SCOUNT=128

sweep \
  proteins_data.csv/train.parquet \
  "parquet" \
  protein_tokens.json \
  --test proteins_data.csv/test.parquet \
  --valid proteins_data.csv/valid.parquet \
  --hyperparameter_sweep params.json.1 \
  --entity_name tyagilab \
  --project_name p_sweep \
  --group_name vocab_32000 \
  --output_dir p_sweep_32000 \
  --vocab_size 32000 \
  --label_names "labels" \
  --model distilbert \
  --sweep_count 128 \
  --metric_opt "eval/f1"

date
exit 0
EOT
