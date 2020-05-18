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



def f_gen_imgs(generator, noise_vector_length, num_images=100, fname=None, multichannel=False):
    """Plots a grid of generated images"""
    samples = generator.predict(np.random.normal(size=(num_images,1,noise_vector_length)))
    if multichannel:
        samples = np.take(samples,0,axis=channel_axis) # take the scaled channel 
    else:
        samples = np.squeeze(samples)
        
    np.save(fname,samples)

if __name__ == '__main__':
    config = 'base'
    runId = '1'
    run = './expts/'+config+'/run'+runId+'/'
    modelpath = run+'models/g_cosmo_best.h5'
    
#     args=f_parse_args()
#     modelpath=args.fname
    
#     main_dir='/global/project/projectdirs/dasrepo/vpa/cosmogan/data/computed_data/exagan1/run_100k_samples_35epochs/models/'
#     modelpath=main_dir+'g_cosmo0029.h5'
    main_dir='/global/cfs/cdirs/dasrepo/vpa/cosmogan/data/computed_data/exagan1/run_200k_samples_40epochs/models/'
    modelpath=main_dir+'g_cosmo_best.h5'
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
    op_fname=main_dir+'gen_imgs.npy'
    num_images=3000
    #print(GAN.noise_vect_len,num_images)
    
    
    #a1=GAN.genrtor.predict(np.random.normal(size=(500,1,64)))
    #print(a1.shape)
    print('Generating images')
    f_gen_imgs(GAN.genrtor, GAN.noise_vect_len, num_images=num_images, fname=op_fname)
    print('Saved images')
    raise SystemExit
    
    # Plot generated images
    plots.save_img_grid(GAN.genrtor, GAN.noise_vect_len, GAN.invtransform, GAN.C_axis, Xterm=True, 
                        scale=GAN.cscale, multichannel=GAN.multichannel)

    # Plot pixel intensity histogram and calculate chi-square score
    chi = plots.pix_intensity_hist(GAN.val_imgs, GAN.genrtor, GAN.noise_vect_len, 
                                   GAN.invtransform, GAN.C_axis, multichannel=GAN.multichannel, Xterm=True)

    # Plot power spectrum and calculate chi-square score
    pschi = plots.pspect(GAN.val_imgs, GAN.genrtor, GAN.invtransform, GAN.noise_vect_len, 
                        GAN.C_axis, Xterm=True, multichannel=GAN.multichannel)

    plt.show()

