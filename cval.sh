#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=2000.cval
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64000
#SBATCH --output=2000.cval.out
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

cross_validate \
  prot.2000.512/train.parquet \
  "parquet" \
  --test prot.2000.512/test.parquet \
  --valid prot.2000.512/valid.parquet \
  --entity_name tyagilab \
  --project_name p_sweep \
  --group_name cval \
  --label_names "labels" \
  --config_from_run tyagilab/prot/lm52lquq \
  --output_dir cval_results \
  --overwrite_output_dir
  

date
exit 0
EOT
