#!/bin/sh -l
#SBATCH -A abuganza
#SBATCH --job-name=single_max
#SBATCH --nodes=2
#SBATCH --ntasks=180
#SBATCH --time=7-00:00:00

module load anaconda
module load hyper-shell
module load use.own
module load conda-env/R21_env_cluster-py3.12.5

hyper-shell cluster inverse_single_max_tasks.txt -N180 -f tasks.failed