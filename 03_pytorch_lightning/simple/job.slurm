#!/bin/bash
#SBATCH --job-name=lightning     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=20       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=00:02:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

module purge
module load bsc/1.0
module load singularity/3.11.5

srun singularity exec --nv ../../pytorch.sif  python example.py
