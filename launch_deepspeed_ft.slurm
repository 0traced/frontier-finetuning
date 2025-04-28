#!/bin/bash

#SBATCH -A TRN040
#SBATCH -J finetune_llama3_ds 
#SBATCH -o logs/finetune_llama3_ds-%j.o
#SBATCH -e logs/finetune_llama3_ds-%j.e
#SBATCH -t 01:00:00
#SBATCH -p batch
##SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --gres=gpu:8
#SBATCH -N 1 


source /gpfs/wolf2/olcf/trn040/world-shared/sajal/miniconda3/bin/activate
module reset
module load rocm/6.1.3
module load gcc/12.2.0
module load miniforge3/23.11.0

export ROCM_HOME=/opt/rocm-6.1.3

conda activate /gpfs/wolf2/olcf/trn040/world-shared/sajal/env-ft-6.1.3-ro
module unload miniforge3/23.11.0
export HF_HUB_OFFLINE=1

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

export HF_TOKEN="YOUR_HF_TOKEN"

scontrol show hostnames $SLURM_NODELIST > job.node.list.$SLURM_JOB_ID
input="./job.node.list".$SLURM_JOB_ID
host_file=host_file.$SLURM_JOB_ID
readarray -t arr <"$input"

for item in "${arr[@]}"; do
  echo "$item" slots=8 >> $host_file 
done

first=${arr[0]}
echo "first=" $first
ips=`ssh $first hostname -I`
read -ra arr <<< ${ips}
export MASTER_ADDR=${arr[0]}
echo "MASTER_ADDR=" $MASTER_ADDR

#torchrun --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=$MASTER_ADDR \
#--master_port=29500 sft_llama_deepspeed.py


#deepspeed --num_gpus 8 --num_nodes 1 --hostfile $host_file --master_addr=$MASTER_ADDR --master_port=29500 \
deepspeed ./sft_llama_ds.py
