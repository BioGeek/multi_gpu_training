# Multi-GPU Training with PyTorch: Data and Model Parallelism

## About
The material in this repo demonstrates multi-GPU training using PyTorch. Part 1 covers how to optimize single-GPU training. The necessary code changes to enable multi-GPU training using the data-parallel and model-parallel approaches are then shown. This workshop aims to prepare researchers to use the new H100 GPU nodes as part of the [MareNostrum 5](https://www.bsc.es/supportkc/docs/MareNostrum5/intro/) supercomputer.

## Singularity Containers

The login nodes on MareNostrum 5 are the only nodes accessible from external networks, and **no connections from the cluster to the outside world are permitted for security reasons.**

When working with a supercomputer that is disconnected from the internet, it becomes challenging to install and manage software dependencies. [Singularity containers](https://docs.sylabs.io/guides/3.5/user-guide/index.html) provide a solution to this problem.

A Singularity container is a lightweight, portable, and self-contained environment that encapsulates all the necessary software dependencies, libraries, and configurations required to run an application. It allows you to package your entire application stack, including the operating system, into a single file.

Advantages include:

 * **Isolation**: Singularity containers provide isolation between the host system and the application. This isolation ensures that the application runs consistently, regardless of the underlying system configuration. It prevents conflicts between different software versions and libraries.

 * **Portability**: Singularity containers are highly portable. You can create a container on one system and run it on another without worrying about compatibility issues. This portability is especially useful when transferring work between different supercomputers or sharing code with collaborators.

* **Reproducibility**: Singularity containers enable reproducibility by capturing the exact software environment in which an application was developed and tested. This ensures that the application behaves consistently across different systems and eliminates the "works on my machine" problem.

* **Ease of deployment**: Singularity containers simplify the deployment process. Instead of manually installing and configuring software dependencies on the supercomputer, you can simply transfer the container file and run it. This saves time and effort, especially when dealing with complex software stacks.

* **Security**: Singularity containers provide a secure execution environment. Since the supercomputer is disconnected from the internet, there is a reduced risk of security vulnerabilities. By using Singularity containers, you can ensure that the application runs in an isolated environment without exposing the host system to potential threats.

## Workflow

Workflow at MareNostrum 5 (similar as the workflow described in [Exploitation of the MareNostrum 4 HPC using ARC-CE](https://www.epj-conferences.org/articles/epjconf/abs/2021/05/epjconf_chep2021_02021/epjconf_chep2021_02021.html))

 * We copy all the input files using by mounting a sshfs file system between the your local laptop and the supercomputer
 * We submit the jobs using the login nodes.
 * The jobs run on validated Singularity images with all the software and data preloaded.
 * We check the status of the jobs using the login nodes.
 * We retrieve the output files using the sshfs filesystem.

## Setup

 1. [Install singularity](https://docs.sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps) locally on your laptop 


 2. Create a directory inside your local machine that will be used as a mount point.

```bash
mylaptop$> mkdir ~/marenostrum5
```

 3. Clone this repository

```bash
mylaptop$> cd ~/marenostrum5
mylaptop$> git clone https://github.com/BioGeek/multi_gpu_training.git
```

 4. Build the pytorch and Jax Singularity containers locally

```bash
mylaptop$> make build_pytorch
mylaptop$> make build_jax
```

  5. Download the MNIST data for [`01_single_gpu`](./01_single_gpu/) and [`02_pytorch_ddp`](./02_pytorch_ddp/)

```bash
mylaptop$> cd 01_single_gpu
mylaptop$> python download_data.py
mylaptop$> cd ..
```

 6. Download the CIFAR-10 data for [`03_pytorch_lightning`](./03_pytorch_lightning/)

```bash
mylaptop$> cd 03_pytorch_lightning
mylaptop$> python download_cifar10.py
mylaptop$> cd ..
```

 7. Dow,load the CodeLlama-7b data for [`04_model_parallel_with_fsdp`](./04_model_parallel_with_fsdp/)


```bash
mylaptop$> cd 04_model_parallel_with_fsdp
mylaptop$> python download_models.py
mylaptop$> cd ..
```

 8. Mount your GPFS home directory on the supercomputer 


```bash
mylaptop$> sshfs -o workaround=rename {username}@transfer1.bsc.es: ~/marenostrum5
```

From now on, you can access that directory. If you access it, you should see your home directory of the GPFS filesystem. Any modifications that you do inside that directory will be replicated to the GPFS filesystem inside the HPC machines.

Inside that directory, you can call "git clone", "git pull" or "git push" as you please.


 9. You can also use `rsync` to sync a local folder with your mounted GPFS home directory

 ```bash
 rsync  --owner --group --chown  {userID}:{grouID} --archive --verbose --update --progress --delete local_folder/ ~/marenostrum5/
 ```

Meaning of some of the flags:

```bash
--archive, -a            archive mode 
--update, -u             skip files that are newer on the receiver
--verbose, -v            increase verbosity
--progress               show progress during transfer
```

Note : If you add a `/` after `local_folder` it means "the contents of `local_folder`".  Without the trailing slash, `rsync` will place `local_folder`, including the directory, within `~/marenostrum5/`. This will create a hierarchy that looks like:

`~/marenostrum5/local_folder`

 10. Or you can use `rsync` to sync a local folder via the transfer node

 ```bash
rsync --owner --group --chown  {userID}:{grouID} -avz --progress --ignore-existing multi_gpu_training {username}@transfer1.bsc.es:"~/"
```



### Authorship

The [original version of this guide](https://github.com/PrincetonUniversity/multi_gpu_training) was created by Mengzhou Xia, Alexander Wettig and Jonathan Halverson. Members of Princeton Research Computing made contributions to this material.

This version has been adapted for the MareNostrum 5 supercomputer by Jeroen Van Goey
