#!/bin/bash
# Interpreter declaration

#SBATCH -A birthright
#SBATCH -p burst 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -J gen_MC_SQ
#SBATCH --mem=8g
#SBATCH -t 10:00:00
#SBATCH -o ./test-output.txt
#SBATCH -e ./test-error.txt
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tungc@ornl.gov
# sbatch arguments

module load anaconda3
conda activate test_CADES

./gen_MC_SQ.sh