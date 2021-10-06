#!/bin/bash
#SBATCH --job-name=gtex-as    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=aj13@princeton.edu

module purge
module load anaconda3
conda activate torch-env

srun python gtex_find_pairwise_as.py
