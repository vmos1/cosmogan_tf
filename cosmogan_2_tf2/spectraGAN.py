from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization,\
                                           ReLU, LeakyReLU, Reshape, Activation, Flatten, Input
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy import fftpack


def azimAvg_tensor(image, center=None):
    """
    Calculate the azimuthally averaged power spectrum (1D), for a batch of image tensors.
    image - The image tensor, [N,C,H,W] format
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    
    """
    batch, channel, height, width = image.shape.as_list()
    # Calculate the indices from the image
    y, x = np.indices([height, width])
    y = np.tile(y, (batch, channel, 1, 1))
    x = np.tile(x, (batch, channel, 1, 1))
    
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = tf.argsort(tf.reshape(r, (batch, channel, -1,)))
    r_sorted = tf.gather(tf.reshape(r, (batch, channel, -1,)), ind, batch_dims=2)
    i_sorted = tf.gather(tf.reshape(image, (batch, channel, -1,)), ind, batch_dims=2)

    
    # Get the integer part of the radii (bin size = 1)
    r_int = tf.cast(r_sorted, tf.int32)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[:,:,1:] - r_int[:,:,:-1]  # Assumes all radii represented
    rind = tf.reshape(tf.where(deltar)[:,2], (batch, -1))    # location of changes in radius
    rind = tf.expand_dims(rind, axis=1)
    nr = tf.cast(rind[:,:,1:] - rind[:,:,:-1], tf.float32)        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    
    csum = tf.cumsum(i_sorted, axis=-1)
    tbin = tf.gather(csum, rind[:,:,1:], batch_dims=2) - tf.gather(csum, rind[:,:,:-1], batch_dims=2)
    radial_prof = tbin / nr

    return radial_prof


@tf.function
def batch_power_spectrum(image):
    """Computes azimuthal average of 2D power spectrum of a batch of images, 
       for use in generator loss function."""
    GLOBAL_MEAN = 1. 
    image = (image - GLOBAL_MEAN)/GLOBAL_MEAN
    shuffled = tf.transpose(tf.cast(image, dtype=tf.complex64), perm=(0,3,1,2))
    F1 = tf.signal.fft2d(shuffled)
    F2 = tf.signal.fftshift(F1, axes=(2,3))
    pspec2d = tf.abs(F2)**2
    P_k = azimAvg_tensor(pspec2d)
    return tf.squeeze(P_k)


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged power spectrum (1D), for a batch of images.
    image - The image tensor, [N,C,H,W] format
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    
    """
    batch, channel, height, width = image.shape
    # Calculate the indices from the image
    y, x = np.indices([height, width])
    
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])
    ind = np.argsort(r.flat)

    # Get sorted radii
    r_sorted = r.flat[ind]
    i_sorted = np.reshape(image, (batch, channel, -1,))[:,:,ind]
    
    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(np.int32)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    
    csum = np.cumsum(i_sorted, axis=-1)
    tbin = csum[:,:,rind[1:]] - csum[:,:,rind[:-1]]
    radial_prof = tbin / nr

    return radial_prof


def power_spectrum(image):
    """Computes azimuthal average of 2D power spectrum of a np array image batch.
       For plotting power spectra of images against validation set."""
    GLOBAL_MEAN = 1. 
    F1 = fftpack.fftn((image - GLOBAL_MEAN)/GLOBAL_MEAN, axes=[1,2])
    F2 = fftpack.fftshift(F1, axes=[1,2])
    pspec2d = np.abs(F2)**2
    pspec2d = np.moveaxis(pspec2d, 3, 1)
    P_k = np.squeeze(azimuthalAverage(pspec2d))
    k = np.arange(P_k.shape[1])
    return k, P_k




class spectraGAN():
    '''
    Class for GAN objects
    init_params -: Initializes the parameters
    
    
    '''

    def __init__(self, expDir):
        # Load parameters
        self.init_params()
        self.expDir = expDir
        
        # Build networks
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(self.learn_rate, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.learn_rate, beta_1=0.5)

        # Load data
        self.real_imgs = np.load(self.dataname+'large_dataset_train.npy').astype(np.float32)
        #self.real_imgs = np.load(self.dataname+'raw_train.npy').astype(np.float32)
        self.real_imgs = self.transform(self.real_imgs) # normalize data using nonlinear transformation
        print(self.real_imgs.shape)
        self.val_imgs = np.load(self.dataname+'large_dataset_val.npy').astype(np.float32)[:3000]
        #self.val_imgs = np.load(self.dataname+'raw_val.npy').astype(np.float32)
        print(self.val_imgs.shape)

        # Setup utils (checkpointing and tensorboard)
        self.checkpoint_dir = os.path.join(expDir,'training_checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(expDir, 'logs'))
        self.prep_summaries()
        

    def init_params(self):
        
        
        ### Code to read from .yaml. 
        ##params = load_params('./config.yaml', configtag)
        ##logging.info(str(params))
        ##self.dataname = params['dataname']
        
        # Hyperparameters
        self.learn_rate = 2E-4
        self.noise_vect = 64
        self.batchsize = 64
        self.Nconvfilters = [64, 128, 256, 512]
        self.Ndeconvfilters = [256, 128, 64, 1]
        self.dataname = '/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/'
        self.checkpoint_prefix = 'ckpt'
        self.tar_Pk_file= 'data/Pk_mean_real.npy'
        self.tar_Pk_std_file= 'data/Pk_std_real.npy'
        
        # Loss weight terms -- can set any of these to 0 to test different combinations
        # of spectral constraint loss, feature matching loss, and R1 regularization.
        # Surprisingly, so far the most effective configuration seems to be just setting all
        # weights to 1.
        self.LAMBDA_1 = 1. # weight term for the spectral constraint on the mean P(k) per k bin
        self.LAMBDA_2 = 1. # weight term for the spectral constraint on the variance of P(k) per k bin
        self.LAMBDA_3 = 1. # weight term for the feature matching loss
        self.R_LAMBDA = 1. # R1 regularization weight
        
        # Load the target P(k) distribution: mean and variance per k bin over training set
        self.tar_Pk = tf.constant(np.load(self.tar_Pk_file).astype(np.float32))
        self.tar_Pk_std = tf.constant(np.load(self.tar_Pk_std_file).astype(np.float32))
        self.tar_Pk_var = tf.pow(self.tar_Pk_std, 2.)


    def prep_summaries(self):
        self.G_loss = tf.keras.metrics.Mean(name='G_loss', dtype=tf.float32)
        self.G_loss_gan = tf.keras.metrics.Mean(name='G_loss_gan', dtype=tf.float32)
        self.G_loss_spect = tf.keras.metrics.Mean(name='G_loss_spect', dtype=tf.float32)
        self.G_loss_spect_var = tf.keras.metrics.Mean(name='G_loss_spect_var', dtype=tf.float32)
        self.D_loss = tf.keras.metrics.Mean(name='D_loss', dtype=tf.float32)


    def Generator(self):
        gmodel = tf.keras.models.Sequential(name="Generator")
        gmodel.add(Dense(8*8*2*self.Ndeconvfilters[0], input_shape=(self.noise_vect,),
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
        gmodel.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
        gmodel.add(ReLU())
        gmodel.add(Reshape((8,8,2*self.Ndeconvfilters[0])))
        for lyrIdx in range(4):
             gmodel.add(Conv2DTranspose(self.Ndeconvfilters[lyrIdx], 5, strides=2, padding='same',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
             if lyrIdx == 3:
                gmodel.add(Activation('tanh')) # last layer has tanh activation
             else:
                gmodel.add(BatchNormalization(epsilon=1e-5, momentum=0.9))
                gmodel.add(ReLU())
        gmodel.summary()
        return gmodel


    def Discriminator(self):
        dmodel = tf.keras.models.Sequential()
        img = Input(shape=[128,128,1])
        x = img
        feats = []
        for lyrIdx in range(4):
            x = Conv2D(filters=self.Nconvfilters[lyrIdx], kernel_size=5, strides=2, padding='same',
                       kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02))(x)
            feats.append(x) # Extract the feature maps immediately following convolutions (before BatchNorm)
            x = BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
            x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        out = Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
        # Discriminator outputs are the 'out' tensor, which is the binary indicator of real/fake,
        # as well as the 'feats' list which is a list of the intermediate activations/feature maps
        # of the discriminator, which are used by the generator in the feature matching loss term. 
        outs = feats + [out]
        disc = tf.keras.Model(inputs=img, outputs=outs, name='Discriminator')
        disc.summary()
        return disc

    
    def generator_loss(self, real_output, fake_output, images):
        # Adversarial loss
        gan = self.loss(tf.ones_like(fake_output[-1]), fake_output[-1])
        
        # Feature matching loss
        FM = 0.
        for i in range(4):
            # take mean of feature maps across N,H,W dimensions (per-channel mean over the batch)
            real_mean = tf.reduce_mean(real_output[i], axis=[0,1,2])
            fake_mean = tf.reduce_mean(fake_output[i], axis=[0,1,2])
            FM = FM + tf.reduce_sum(tf.square(real_mean - fake_mean))

        # Spectral loss: calculate P(k) for the batch of generated images
        images = self.invtransform(images) # transform images back to original data units
        gen_Pk = batch_power_spectrum(images) # Calculate 1D power spectrum for each image
        gen_mean, gen_var = tf.nn.moments(gen_Pk, axes=0) # Compute mean and variance for P(k) per k bin
        
        # Only enforce spectral loss on first 64 k bins in P(k) because k>64 corresponds to sub-pixel scale in 128x128 images
        # spectral loss is log mean-squared error across all k bins for batchwise mean and variance of P(k)
        spectmean = tf.math.log(tf.reduce_mean(tf.pow(gen_mean[:64] - self.tar_Pk[:64], 2.)))
        spectvar = tf.math.log(tf.reduce_mean(tf.pow(gen_var[:64] - self.tar_Pk_var[:64], 2.)))
        
        return gan + self.LAMBDA_1*spectmean + self.LAMBDA_2*spectvar + self.LAMBDA_3*FM, gan, spectmean, spectvar


    def discriminator_loss(self, real_output, fake_output, grads):
        rand = tf.random.uniform(minval=0., maxval=1., shape=())
        real_labels = tf.case([(tf.math.greater(rand, 0.01), lambda: tf.ones_like(real_output))],
                              default = lambda: tf.zeros_like(real_output), exclusive = True)
        fake_labels = tf.case([(tf.math.greater(rand, 0.01), lambda: tf.zeros_like(fake_output))],
                              default = lambda: tf.ones_like(fake_output), exclusive = True)
        real_loss = self.loss(real_labels, real_output)
        fake_loss = self.loss(fake_labels, fake_output)
        # R1 regularization: enforce gradient penalty on real images
        R1term = tf.reduce_mean(tf.reduce_sum(tf.square(grads), axis=[1,2,3]))
        return real_loss + fake_loss + self.R_LAMBDA*R1term


    @tf.function
    def train_step(self, samples):
        '''
        Steps:
        - Generate fake images from noise using Generator
        - Generate real output and fake output running the Discriminator
        - Compute the generator loss 
        - Compute gradients on real data
        - Compute discriminator loss
        - Compute generator and gradient loss
        - Apply optimizer to generator and discriminator using gradients
        '''
        
        noise = tf.random.normal([self.batchsize, self.noise_vect])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape(persistent = True) as disc_tape:
            disc_tape.watch(samples)
            # Generate images using generator
            generated = self.generator(noise, training=True)
            # Get labels of real and fake images using discriminator
            real_output = self.discriminator(samples, training=True)
            fake_output = self.discriminator(generated, training=True)
            # compute generator loss 
            gen_loss, gan_loss, spect_loss, spect_var_loss = self.generator_loss(real_output, fake_output, generated)
            # Compute gradients of discriminiator to use in discriminator loss
            real_grads = disc_tape.gradient(real_output[-1], samples)
            # Compute discriminator loss
            disc_loss = self.discriminator_loss(real_output[-1], fake_output[-1], real_grads)
        
        # Update state of losses for tensorboard monitoring
        self.G_loss.update_state(gen_loss)
        self.G_loss_gan.update_state(gan_loss)
        self.G_loss_spect.update_state(spect_loss)
        self.G_loss_spect_var.update_state(spect_var_loss)
        self.D_loss.update_state(disc_loss)

        # Apply gradient updates
        # compute generator gradients using (loss,vars)
        g_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        # compute discriminator loss
        d_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        # use gradients on optimizer
        self.generator_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        del disc_tape


    def transform(self, x):
        return 2.*x/(x + 4.) - 1.


    def invtransform(self, s):
        return 4.*(1. + s)/(1. - s)


    def generate_images(self):
        imgs = self.generator(tf.random.normal(shape=(4, self.noise_vect)), training = False)
        fig = plt.figure(figsize=(6,6))
        for i in range(4):
            plt.subplot(2,2,i+1)
            img = imgs.numpy()[i,:,:,0]
            plt.imshow(img, cmap='Blues')
            plt.axis('off')
        return fig


    def pix_hist(self):
        num = 1000
        imgs = self.generator(tf.random.normal(shape=(num, self.noise_vect)), training = False)
        imgs = self.invtransform(imgs.numpy())
        vals = self.val_imgs[np.random.randint(len(self.val_imgs), size=(num))]
        val_hist, bin_edges = np.histogram(vals, bins=50)
        gen_hist, _ = np.histogram(imgs, bins=bin_edges)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig = plt.figure()
        plt.errorbar(centers, val_hist, yerr=np.sqrt(val_hist), fmt='ks--', label='real')
        plt.errorbar(centers, gen_hist, yerr=np.sqrt(gen_hist), fmt='ro', label='generated')
        plt.xlabel('Dark Matter Density')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.legend()
        sqdiff = np.power(val_hist - gen_hist, 2.0)
        val_hist[val_hist<=0.] = 1.
        return (fig, np.sum(np.divide(sqdiff, val_hist)))


    def pspect(self):
        '''
        Compute 
        '''
        num = 1000
        imgs = self.generator(tf.random.normal(shape=(num, self.noise_vect)), training = False)
        imgs = self.invtransform(imgs.numpy())
        vals = self.val_imgs[np.random.randint(len(self.val_imgs), size=(num))]
        k, Pk_val = power_spectrum(vals)
        k, Pk_gen = power_spectrum(imgs)
        val_mean = np.mean(Pk_val, axis=0)
        gen_mean = np.mean(Pk_gen, axis=0)
        val_std = np.std(Pk_val, axis=0)
        gen_std = np.std(Pk_gen, axis=0)


        fig = plt.figure()
        plt.fill_between(k, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.4)
        plt.plot(k, val_mean, 'k:')
        plt.plot(k, gen_mean, 'r--')
        plt.plot(k, val_mean + val_std, 'k-')
        plt.plot(k, val_mean - val_std, 'k-')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(r'$P(k)$')
        plt.xlabel(r'$k$')
        plt.title('Power Spectrum')
        return (fig, np.sum(np.divide(np.power(gen_mean[:64] - val_mean[:64], 2.0), val_mean[:64])))

