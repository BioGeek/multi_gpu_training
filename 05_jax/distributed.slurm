#!/bin/bash
#SBATCH --job-name=jax_distributed  # Job name
#SBATCH --nodes=4                   # Number of nodes
#SBATCH --ntasks=32                 # Total number of tasks
#SBATCH --ntasks-per-node=8         # Number of tasks per node
#SBATCH --gpus-per-node=8           # Number of GPUs per node
#SBATCH --time=00:01:00        # Time limit hrs:min:sec (example)
#SBATCH --output=job_%j.log    # Standard output and error log

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

# distribute the python script to all nodes
time srun singularity exec --nv ../jax.sif python distributed.py

# Get final ACC budget
final_budget=$(parse_acc_budget)
echo "Final ACC budget: $final_budget"

# Calculate and report spent budget
spent_budget=$(echo "$initial_budget - $final_budget" | bc)
echo "Spent compute budget: $spent_budget"
