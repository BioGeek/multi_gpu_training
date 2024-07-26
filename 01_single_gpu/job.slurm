#!/bin/bash
#SBATCH --job-name=mnist         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=20        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

module purge
module load bsc/1.0
module load singularity/3.11.5

# Function to parse ACC budget
parse_acc_budget() {
        bsc_acct | awk '/CPU GROUP BUDGET:/,/USER CONSUMED CPU:/' | grep 'Marenostrum5 ACC' | awk '{print $3}'
}

# Get initial ACC budget
initial_budget=$(parse_acc_budget)
echo "Initial ACC budget: $initial_budget"

# which gpu node was used
echo "Running on host" $(hostname)

# print the slurm environment variables sorted by name
printenv | grep -i slurm | sort

# Set default value for num_workers if not provided
: "${num_workers:=1}"

# Print the number of workers being used
echo "Number of workers: $num_workers"

# Use the num_workers environment variable, defaulting to 1 if not set
time singularity run --nv ../pytorch.sif kernprof -o ${SLURM_JOBID}_num_workers_${num_workers}.lprof -l mnist_classify.py --epochs=3 --num-workers=$num_workers

# Get final ACC budget
final_budget=$(parse_acc_budget)
echo "Final ACC budget: $final_budget"

# Calculate and report spent budget
spent_budget=$(echo "$initial_budget - $final_budget" | bc)
echo "Spent compute budget: $spent_budget"
