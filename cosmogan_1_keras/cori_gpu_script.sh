#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --qos=regular
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -c 8 
#SBATCH --job-name=exagan_keras_train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=gpu
#SBATCH --account=m3363
#SBATCH --gres=gpu:1
#################


echo "--start date" `date` `date +%s`
echo '--hostname ' $HOSTNAME
export HDF5_USE_FILE_LOCKING=FALSE
module unload esslurm
module load tensorflow/gpu-1.15.0-py37
### Actual script to run
srun python train.py base 2
echo "--end date" `date` `date +%s`
