# Multi-GPU Training with PyTorch: Data and Model Parallelism

### About
The material in this repo demonstrates multi-GPU training using PyTorch. Part 1 covers how to optimize single-GPU training. The necessary code changes to enable multi-GPU training using the data-parallel and model-parallel approaches are then shown. This workshop aims to prepare researchers to use the new H100 GPU nodes as part of Princeton Language and Intelligence.

### Setup

Make sure you can run Python on MareNostrum 5:

```bash
mylaptop$> ssh {username}@glogin1.bsc.es # General purpuse login node. use `alogin1.bsc.es` for the GPU login node
[{username}@glogin1 ~]$ module load anaconda3/2024.02
load ANACONDA/2024.02 (PATH)
[{username}@glogin1 ~]$ which python
/apps/ACC/ANACONDA/2024.02/bin/python
[{username}@glogin1 ~]$ python --version
Python 3.11.7
```

The login nodes are the only nodes accessible from external networks, and **no connections from the cluster to the outside world are permitted for security reasons.**

> [!WARNING] 
> All file transfers from/to the outside must be executed from your local machine and **not within the cluster.**

In general, to copy files or directories from an external machine to MareNostrum 5:

```bash
mylaptop$> scp -r "mylaptop_SOURCE_dir" {username}@transfer1.bsc.es:"MN5_DEST_dir"
```

So to use this repository in MareNostrum v5

```bash
mylaptop$> git clone https://github.com/BioGeek/multi_gpu_training.git
mylaptop$> scp -r "multi_gpu_training" {username}@transfer1.bsc.es:"~/multi_gpu_training"
```


### Authorship

The [original version of this guide](https://github.com/PrincetonUniversity/multi_gpu_training) was created by Mengzhou Xia, Alexander Wettig and Jonathan Halverson. Members of Princeton Research Computing made contributions to this material.

Tis version has been adapted for the MareNostrum 5 supercomputer by Jeroen Van Goey
