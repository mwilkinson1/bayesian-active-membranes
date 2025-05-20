#!/bin/sh -l
#SBATCH -A abuganza
#SBATCH --job-name=InVivoParamSweep
#SBATCH --nodes=1
#SBATCH --ntasks=92
#SBATCH --time=1-00:00:00

module load intel/19.1.3.304 
module load abaqus
module load anaconda
module load hyper-shell

python input_generator.py

hyper-shell cluster master_task_file.in -N16 -f tasks.failed

python consolidate_data.py