#!/bin/bash

module load rocm/6.1.3
module load gcc/12.2.0
module load miniforge3/23.11.0

export ROCM_HOME=/opt/rocm-6.1.3

#source /gpfs/wolf2/olcf/trn040/world-shared/sajal/miniconda3/bin/activate
conda activate /gpfs/wolf2/olcf/trn040/world-shared/sajal/env-ft-6.1.3-ro
module unload miniforge3/23.11.0
export HF_HUB_OFFLINE=0


