# PyTorch Lightning

[PyTorch Lightning](https://www.pytorchlightning.ai) wraps PyTorch to provide easy, distributed training done in your choice of numerical precision for the matrix multiples and other operations. To convert from PyTorch to PyTorch Lightning one simply needs to:

+ restructure the code by moving the network definition, optimizer and other code to a subclass of `L.LightningModule`  
+ remove `.cuda()` and `.to()` calls since Lightning code is hardware agnostic  

Once these changes have been made one can simply choose how many nodes or GPUs to use and Lightning will take care of the rest. One can also use different numerical precisions (fp16, bf16). There is tensorboard support and [model-parallel training](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html).

See the [Trainer API](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api).

## Multi-GPU Example

Let's work through this [example](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/cifar10-baseline.html) where a modified ResNet-18 model is trained on [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10). Here is the application script:

```python
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

seed_everything(7)

BATCH_SIZE = 256
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
NUM_WORKERS = int(os.environ["SLURM_CPUS_PER_TASK"])
NUM_NODES = int(os.environ["SLURM_NNODES"])
ALLOCATED_GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

def create_model():
    model = torchvision.models.resnet18(weights=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=False)
            self.log(f"{stage}_acc", acc, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

model = LitResnet(lr=0.05)
model.datamodule = cifar10_dm

trainer = Trainer(
    devices=ALLOCATED_GPUS_PER_NODE,
    accelerator="gpu",
    num_nodes=NUM_NODES,
    strategy='ddp',
    precision=32,
    max_epochs=10)

trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)
```

### Step 1: Download the data

The compute nodes do not have internet access so download the data on the login node:

```
$ cd multi_gpu_training/03_pytorch_lightning/multi
(torch-env)$ python download_cifar10.py
```

### Step 2: Submit the Job

Below is the Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:2             # number of allocated gpus per node
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

export MASTER_PORT=$(get_free_port)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# distribute the python script to all nodes
time srun singularity exec --nv ../pytorch.sif python mnist_classify_ddp.py --epochs=2

# Get final ACC budget
final_budget=$(parse_acc_budget)
echo "Final ACC budget: $final_budget"

# Calculate and report spent budget
spent_budget=$(echo "$initial_budget - $final_budget" | bc)
echo "Spent compute budget: $spent_budget"
```

Submit the job:

```bash
[{username}@alogin1 03_pytorch_lightning]$ sbatch --account={account} --qos={qos}  job.slurm
Submitted batch job {slurm_jobid}
```

How does the training time decrease in going from 2 to 1 GPUs? What happens if you use `precision=16`?

## Numerical Precision

You can try adjusting the `precision` to accelerate training at the expense of accuracy. See the options for `precision` [here](https://lightning.ai/docs/pytorch/latest/common/trainer.html#trainer-class-api).

## Debugging

For troubleshooting NCCL try adding these environment variables to your Slurm script:

```
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

## Useful Links

+ [PyTorch Lightning and Slurm](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html)  
+ [PyTorch LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.html)


