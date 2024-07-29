#!/bin/bash -l
#SBATCH --job-name=chess_finetune  # create a short name for your job
#SBATCH --output=%x-%j_%n.out      # file to write stdout
#SBATCH --nodes=1                  # node count
#SBATCH --ntasks=1                 # total number of tasks across all nodes
#SBATCH --cpus-per-task=80         # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4               # number of gpus per node
#SBATCH --time=01:00:00            # total run time limit (HH:MM:SS)

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

# You can override the default values of these parameters by adding `TOTAL_BATCH_SIZE=... sbatch chess_finetune.sh`
# The default values are given after the minus (i.e., 64 and 1) and are used if the variable is empty
total_batch_size=${TOTAL_BATCH_SIZE:-64} # total batch size per optimization
batch_size_per_device=${BATCH_SIZE_PER_DEVICE:-1} # batch size per device

# Read CUDA_VISIBLE_DEVICES to detect the number of GPUs
num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")

# Compute the gradient accumulation steps given a total batch size and a batch size per device
gradient_accumulation_steps=$(($total_batch_size / $batch_size_per_device / $num_gpus))

time singularity exec --nv ../../pytorch.sif torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc-per-node=$num_gpus \
    chess_finetune.py \
        --batch_size_per_device $batch_size_per_device \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        $@

# Get final ACC budget
final_budget=$(parse_acc_budget)
echo "Final ACC budget: $final_budget"

# Calculate and report spent budget
spent_budget=$(echo "$initial_budget - $final_budget" | bc)
echo "Spent compute budget: $spent_budget"