# Multi-GPU Training with PyTorch: Data and Model Parallelism

## About
The material in this repo demonstrates multi-GPU training using PyTorch. Part 1 covers how to optimize single-GPU training. The necessary code changes to enable multi-GPU training using the data-parallel and model-parallel approaches are then shown. This workshop aims to prepare researchers to use the new H100 GPU nodes as part of Princeton Language and Intelligence.

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


## Setup

 1. [Install singularity](https://docs.sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps) locally on your laptop 
 2. Clone this repository

```bash
mylaptop$> git clone https://github.com/BioGeek/multi_gpu_training.git
```

 3. Build the Singularity container locally

```bash
mylaptop$> make build
```

  4. Download the MNIST data

```bash
mylaptop$> cd 01_single_gpu
mylaptop$> python download_data.py
mylaptop$> cd ..
```

 5. Upload the repository (including data and singularity file) via the MareNostrum transfer node to your home directory on the supercomputer

```bash
mylaptop$> cd .. 
mylaptop$> rsync -avz --progress --ignore-existing multi_gpu_training {username}@transfer1.bsc.es:"~/"
```


### Authorship

The [original version of this guide](https://github.com/PrincetonUniversity/multi_gpu_training) was created by Mengzhou Xia, Alexander Wettig and Jonathan Halverson. Members of Princeton Research Computing made contributions to this material.

This version has been adapted for the MareNostrum 5 supercomputer by Jeroen Van Goey
