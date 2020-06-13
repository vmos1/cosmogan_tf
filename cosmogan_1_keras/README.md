



Two bash submission files: `train_gan.sh` `gen_img.sh`

- train_gan.sh trains a GAN from data

- gen_img.sh needs an input file for a stored GAN model. It uses it to generate 5000 images. The resultant image are stored in the same folder as the model

Run code to generate images
- salloc -A nstaff -N 1 -C gpu --gres=gpu:1 -c 8 -t 00:30:00
- module load tensorflow/gpu-1.15.0-py37
- srun python generate_images.py -f /global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_from_other_code/exagan1/run_1_same_cosmology/models/g_cosmo_best.h5
