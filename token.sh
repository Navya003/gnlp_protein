#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="protein"
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4062
#SBATCH --output=prot.out
#SBATCH --account=lz25
#SBATCH --export=NONE
##SBATCH --partition=comp
#SBATCH --partition=genomics
#SBATCH --qos=genomics
#SBATCH --mail-user=ext-ntyagi@monash.edu
#SBATCH --mail-type=ALL
#======START=====
source /projects/lz25/navyat/conda
conda activate /projects/lz25/navyat/conda/envs/something

date

infile_dna="dna_binding.fa.gz"
infile_rna="rna_binding.fa.gz"



for i in 2000 8000 16000 32000; do
  tokenise_bio \
    -i ${infile_dna} ${infile_rna} \
    -t protein.${i}.json \
    -v ${i}
  create_dataset_bio \
    ${infile_dna} ${infile_rna} protein.${i}.json \
    -o protein.${i}.512 \
    --no_reverse_complement -c 512
done
exit 0
EOT

date
