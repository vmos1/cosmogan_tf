#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --time=00:25:00
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
module load tensorflow/gpu-1.15.0-py37
### Actual script to run
model_file='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_from_other_code/exagan1/run_1_same_cosmology/models/g_cosmo_best.h5'
echo "Using model file" $model_file
srun python generate_images.py -f $model_file

echo "--end date" `date` `date +%s`
