#!/bin/bash
#SBATCH --partition=large-mem
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=NH312PES
#SBATCH --chdir=./
#SBATCH -o NH312PESo.txt
#SBATCH -e NH312PESe.txt

# Changes working directory to the directory where this script is submitted from
printf 'Changing to the working directory: %s\n\n' "$SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

# Load Necessary Modules -- Add whatever modules you need to run your program
#printf 'Loading modules\n'


python NH312PES.py

# Determine the job host names and write a hosts file
srun -n $SLURM_NTASKS hostname | sort -u > $SLURM_JOB_ID.hosts

