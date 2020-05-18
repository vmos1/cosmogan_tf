import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('./utils')
sys.path.append('./networks')
import logging
import logging_utils
import GANbuild
import parameters
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow.keras import backend as K


### Additions for horovod

if __name__=="__main__":
    
    horovod_flag=False
    
    if horovod_flag: 
        print("Using Horovod")
        # Horovod: initialize Horovod.
        hvd.init()
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))
    
    
    configtag = sys.argv[1]
    run_num = sys.argv[2]
    
    # baseDir = '/global/project/projectdirs/dasrepo/vpa/cosmogan/data/'+configtag+'/'
    #baseDir = '/global/project/projectdirs/dasrepo/vpa/cosmogan/data/computed_data/exagan1/'
    baseDir = '/global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_from_other_code/exagan1/'
    
    expDir = baseDir+'run'+str(run_num)+'/'
    if not os.path.isdir(baseDir):
        os.mkdir(baseDir)
    if not os.path.isdir(expDir):
        os.mkdir(expDir)
        os.mkdir(expDir+'models')
    else:
        print("Experiment directory %s already exists, exiting"%expDir)
        sys.exit()

    #Set up logger
    logging_utils.config_logger(log_level=logging.INFO)
    logging_utils.log_to_file(logger_name=None, log_filename=expDir+'train.log')


    # Build GAN
    GAN = GANbuild.DCGAN(configtag, expDir, horovod_flag)

    Nepochs = GAN.Nepochs
    Nbatches = GAN.n_imgs // GAN.batchsize


    for epoch in np.arange(GAN.start, Nepochs+GAN.start):
        logging.info("| ******************************* Epoch %d of %d ******************************* |"%(epoch+1, Nepochs+GAN.start))
        shuff_idxs = np.random.permutation(GAN.n_imgs)
        GAN.train_epoch(shuff_idxs, Nbatches, epoch)

        if (epoch+1)%10==0: 
            GAN.genrtor.save_weights(GAN.expDir+'models/g_cosmo%04d.h5'%(epoch))
            GAN.discrim.save_weights(GAN.expDir+'models/d_cosmo%04d.h5'%(epoch))
    
    
    ### For both Generator and Discriminator, save model and weights
    GAN.genrtor.save_weights(GAN.expDir+'models/g_cosmo_last.h5')
    GAN.discrim.save_weights(GAN.expDir+'models/d_cosmo_last.h5')
    
    logging.info('DONE')



