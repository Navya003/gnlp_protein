#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=2000.train
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64000
#SBATCH --output=2000.train.out
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

train \
  prot.2000.512/train.parquet \
  "parquet" \
  prot.2000.json \
  --test prot.2000.512/test.parquet \
  --valid prot.2000.512/valid.parquet \
  --entity_name tyagilab \
  --project_name prot \
  --group_name train.2000 \
  --label_names "labels" \
  --config_from_run tyagilab/prot/lm52lquq \
  --output_dir train.out \
  --override_output_dir

date
exit 0
EOT
