# Single-GPU Training

It is important to optimize your script for the single-GPU case before moving to multi-GPU training. This is because as you request more resources, your queue time increases. We also want to avoid wasting resources by running code that is not optimized.

Here we train a CNN on the MNIST dataset using a single GPU as an example. We profile the code and make performance improvements.

This tutorial uses PyTorch but the steps are the similar for TensorFlow. See our [TensorFlow](https://researchcomputing.princeton.edu/support/knowledge-base/tensorflow#install) page and the [performance tuning guide](https://tigress-web.princeton.edu/~jdh4/TensorflowPerformanceOptimization_GTC2021.pdf).

## Step 1: Load modules



```bash
mylaptop$> ssh {username}@alogin1.bsc.es # GPU login node. 
[{username}@alogin1 ~]$ module load module load singularity/3.11.5
[{username}@alogin1 ~]$ cd multi_gpu_training
```


Watch a [video](https://www.youtube.com/watch?v=wqTgM-Wq4YY&t=296s) that covers everything on this page for single-GPU training with [profiling Python](https://researchcomputing.princeton.edu/python-profiling) using `line_profiler`.

## Step 2: Run and Profile the Script

First, inspect the script ([see script](mnist_classify.py)) by running these commands:

```bash
[{username}@alogin1 multi_gpu_training]$ cd 01_single_gpu
[{username}@alogin1 01_single_gpu]$ cat mnist_classify.py
```

We will profile the `train` function using `line_profiler`. See line 39 where the `@profile` decorator has been added:

```python
@profile
def train(args, model, device, train_loader, optimizer, epoch):
```

Below is the Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=mnist         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=20       # cpu-cores per task (>1 if multi-threaded tasks)
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

# Print the number of workers being used
echo "Number of workers: ${num_workers:-1}"

# Use the num_workers environment variable, defaulting to 1 if not set
time singularity run --nv ../pytorch.sif kernprof -o ${SLURM_JOBID}.lprof -l mnist_classify.py --epochs=3 --num-workers=${num_workers:-1}

# Get final ACC budget
final_budget=$(parse_acc_budget)
echo "Final ACC budget: $final_budget"

# Calculate and report spent budget
spent_budget=$(echo "$initial_budget - $final_budget" | bc)
echo "Spent compute budget: $spent_budget"

```

[`kernprof`](https://kernprof.readthedocs.io/en/latest/kernprof.html) is a profiler that wraps Python.

Finally, submit the job with your account to the [queue system](https://www.bsc.es/supportkc/docs/MareNostrum5/slurm#queues-qos).

```bash
[{username}@alogin1 01_single_gpu]$ sbatch --account={account} --qos={qos} job.slurm --export=num_workers='1'
Submitted batch job {slurm_jobid}
```

You can display all submitted jobs (from all your current accounts/projects) with `squeue`:

```bash
[deep789424@alogin1 01_single_gpu]$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           {slurm_jobid}       acc    mnist {username}  R       0:12      1 as01r1b16
```

You should find that the code runs in about 23 seconds (some variation in the run time is expected when multiple users are running on the same node. ) and outputs two files:

 * {slurm_jobid}.lprof
 * slurm-{slurm_jobid}.out

Looking into the output, we see we obtained a pretty good accuracy of 99%. For suc a small job, the spent GPU budget is neglible.

```bash
[deep789424@alogin1 01_single_gpu]$ cat slurm-{slurm_jobid}.out

[omitted training log output]
Test set: Average loss: 0.0344, Accuracy: 9893/10000 (99%)

Wrote profile results to {slurm_jobid}.lprof
Inspect results with:
python -m line_profiler -rmt "{slurm_jobid}.lprof"

real	0m23.061s
user	0m24.867s
sys	0m7.626s
Final ACC budget: 2560000.00
Spent compute budget: 0
```


## Step 3: Analyze the Profiling Data

We installed [line_profiler](https://researchcomputing.princeton.edu/python-profiling) into the Singularity container and profiled the code. To analyze the profiling data:

```bash
[{username}@alogin1 01_single_gpu]$ module load singularity/3.11.5 
load SINGULARITY/3.11.5 (PATH)
[{username}@alogin1 01_single_gpu]$ singularity run ../pytorch.sif python -m line_profiler -rmt *.lprof
Timer unit: 1e-06 s

Total time: 15.8417 s
File: mnist_classify.py
Function: train at line 39

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    39                                           @profile                                                                   
    40                                           def train(args, model, device, train_loader, optimizer, epoch):            
    41         3        178.7     59.6      0.0      model.train()                                                          
    42      2817   11691421.4   4150.3     73.8      for batch_idx, (data, target) in enumerate(train_loader):              
    43      2814     217178.5     77.2      1.4          data, target = data.to(device), target.to(device)                  
    44      2814     134220.4     47.7      0.8          optimizer.zero_grad()                                              
    45      2814    1630352.4    579.4     10.3          output = model(data)                                               
    46      2814      99303.6     35.3      0.6          loss = F.nll_loss(output, target)                                  
    47      2814    1103259.3    392.1      7.0          loss.backward()                                                    
    48      2814     948123.0    336.9      6.0          optimizer.step()                                                   
    49      2814       1956.3      0.7      0.0          if batch_idx % args.log_interval == 0:                             
    50       564       1678.7      3.0      0.0              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    51       282       2001.9      7.1      0.0                  epoch, batch_idx * len(data), len(train_loader.dataset),   
    52       282      11930.2     42.3      0.1                  100. * batch_idx / len(train_loader), loss.item()))        
    53       282        119.8      0.4      0.0              if args.dry_run:                                               
    54                                                           break                                                      


 15.84 seconds - mnist_classify.py:39 - train
```

The slowest line is number 42 which consumes 73.8% of the time in the training function. That line involves `train_loader` which is the data loader for the training set. Are you surprised that the data loader is the slowest step and not the forward pass or calculation of the gradients? Can we improve on this?

### Examine Your GPU Utilization

Use tools like [jobstats](https://researchcomputing.princeton.edu/support/knowledge-base/job-stats#jobstats), [gpudash](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#gpudash) and [stats.rc](https://researchcomputing.princeton.edu/support/knowledge-base/job-stats#stats.rc) to measure your GPU utilization. You can also do this on a [compute node in real time](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#gpu-utilization).

Note that GPU utilization as measured using nvidia-smi is only a measure of the fraction of the time that a GPU kernel is running on the GPU. It says nothing about how many CUDA cores are being used or how efficiently the GPU kernels have been written. However, for codes used by large communities, one can generally associate GPU utilization with overall GPU efficiency. For a more accurate measure of GPU utilization, use [Nsight Systems or Nsight Compute](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#profiling) to measure the occupancy.

## Step 4: Work through the Performance Tuning Guide

Make sure you optimize the single GPU case before going to multiple GPUs by working through the [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

## Step 5: Optimize Your Script

One technique that was discussed in the [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) was using multiple CPU-cores to speed-up [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load). Let's put this into practice.

![multiple_workers](https://www.telesens.co/wp-content/uploads/2019/04/img_5ca4eff975d80.png)

*Credit for image above is [here](https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/).*

In `mnist_classify.py`, change `num_workers` from 1 to 8. And then in `job.slurm` change `--cpus-per-task` from 1 to 8. Then run the script again and note the speed-up:

```
[{username}@alogin1 ~]$ sbatch --reservation=multigpu job.slurm
```

How did the profiling data change? Watch the [video](https://www.youtube.com/watch?v=wqTgM-Wq4YY&t=296s) for the solution. For consistency between the Slurm script and PyTorch script, one can use:

```python
import os
...
    cuda_kwargs = {'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
...
```

Several environment variables are set in the Slurm script. These can be referenced by the PyTorch script as demonstrated above. To see all of the available environment variables that are set in the Slurm script, add this line to `job.slurm`:

```
printenv | sort
```

Consider these external data loading libraries: [ffcv](https://github.com/libffcv/ffcv) and [NVIDIA DALI](https://developer.nvidia.com/dali).

## Summary

It is essential to optimize your code before going to multi-GPU training since the inefficiencies will only be magnified otherwise. The more GPUs you request in a Slurm job, the longer you will wait for the job to run. If you can get your work done using an optimized script running on a single GPU then proceed that way. Do not use multiple GPUs if your GPU efficiency is low. The average GPU efficiency on Della is around 50%.

Next, we focus on scaling the code to multiple GPUs (go to [next section](../02_pytorch_ddp)).

