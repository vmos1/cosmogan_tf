import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./networks/')
sys.path.append('./utils/')
import GANbuild
import plots
import argparse

def f_parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run script to train GAN using LBANN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--fname','-f', type=str, default='fname.h5', help='Model file name')

    return parser.parse_args()



def f_gen_imgs(generator, noise_vector_length, num_images, fname, multichannel):
    """Plots a grid of generated images"""
    samples = generator.predict(np.random.normal(size=(num_images,1,noise_vector_length)))
    print(samples.shape)
    channel_axis=1
    if multichannel:
        samples = np.take(samples,0,axis=channel_axis) # take the scaled channel 
    else:
        samples = np.squeeze(samples)
    
    print(samples.shape)
    np.save(fname,samples)

if __name__ == '__main__':
    config = 'base'
    runId = '1'
    run = './expts/'+config+'/run'+runId+'/'
    modelpath = run+'models/g_cosmo_best.h5'
    
    args=f_parse_args()
    modelpath=args.fname
    print(modelpath)
    
    if not os.path.isfile(modelpath):
        print("Error: File %s with pre-trained weights could not be found")
        sys.exit()
    
    
    GAN = GANbuild.DCGAN(config, run, horovod_flag=False)
    print("Compiled model")
    GAN.genrtor.load_weights(modelpath)
    print("Loaded model")
    #print(GAN.genrtor.summary())
    # Generate images
    ### Get folder from the modelpath. Store result in the same folder
    fle=modelpath.split('/')[-1]
    fldr=modelpath.split(fle)[0]
    op_fname=fldr+'gen_imgs.npy'
    num_images=5000
    print('Generating images')
    f_gen_imgs(GAN.genrtor, GAN.noise_vect_len, num_images=num_images, fname=op_fname, multichannel=True)
    print('Saved images')
    

