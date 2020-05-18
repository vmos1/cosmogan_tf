import numpy as np
import os
import sys
import io
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import spectraGAN
import horovod.tensorflow as hvd

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


if __name__=="__main__":

    horovod_flag=False

    if horovod_flag: 
        hvd.init()  ### Initialize horovod
        config=tf.compat.v1.ConfigProto
        print(hvd.local_rank())
        config.gpu_options.visible_device_list=str(hvd.local_rank())
        
    run_num = sys.argv[1]

    continuing = False
    baseDir = '/global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_from_other_code/exagan2/expts/'
    
    expDir = baseDir+'run'+str(run_num)+'/'
    if not os.path.isdir(baseDir):
        os.mkdir(baseDir)
    if not os.path.isdir(expDir):
        os.mkdir(expDir)

    # Build GAN
    GAN = spectraGAN.spectraGAN(expDir)
    t_begin=time.time()
    bestchi = 1e15
    bestPchi = 1e15
    
    num_epochs=80
    for epoch in range(num_epochs):
        start = time.time()
        train_size = len(GAN.real_imgs)
        np.random.shuffle(GAN.real_imgs) 
        with GAN.train_summary_writer.as_default():
            for batchnum in range(train_size//GAN.batchsize):
                samples = GAN.real_imgs[batchnum*GAN.batchsize:(batchnum+1)*GAN.batchsize]
                GAN.train_step(samples)
                if tf.equal(GAN.generator_optimizer.iterations % 100, 0):
                    # Generate sample imgs
                    fig = GAN.generate_images()
                    tf.summary.image("genimg", plot_to_image(fig),
                                     step=GAN.generator_optimizer.iterations)
                    fig, chi = GAN.pix_hist()
                    tf.summary.image("pixhist", plot_to_image(fig),
                                     step=GAN.generator_optimizer.iterations)
                    fig, Pchi = GAN.pspect()
                    tf.summary.image("powerspect", plot_to_image(fig),
                                     step=GAN.generator_optimizer.iterations)

                    # Log scalars
                    tf.summary.scalar('G_loss', GAN.G_loss.result(),
                                      step=GAN.generator_optimizer.iterations)
                    tf.summary.scalar('G_loss_gan', GAN.G_loss_gan.result(),
                                      step=GAN.generator_optimizer.iterations)
                    tf.summary.scalar('G_loss_spect', GAN.G_loss_spect.result(),
                                      step=GAN.generator_optimizer.iterations)
                    tf.summary.scalar('G_loss_spect_var', GAN.G_loss_spect_var.result(),
                                      step=GAN.generator_optimizer.iterations)
                    tf.summary.scalar('D_loss', GAN.D_loss.result(),
                                      step=GAN.generator_optimizer.iterations)
                    tf.summary.scalar('chi', chi, step=GAN.generator_optimizer.iterations)
                    tf.summary.scalar('Pchi', Pchi, step=GAN.generator_optimizer.iterations)
                    GAN.D_loss.reset_states()
                    GAN.G_loss.reset_states()
                    GAN.G_loss_gan.reset_states()
                    GAN.G_loss_spect.reset_states()
                    GAN.G_loss_spect_var.reset_states()

                    # Save model if chi is good
                    if GAN.generator_optimizer.iterations > 10000:
                        if chi < bestchi:
                            GAN.checkpoint.write(file_prefix = os.path.join(GAN.checkpoint_dir, 'BESTCHI'))
                            bestchi = chi
                            print('BESTCHI: iter=%d, chi=%f'%(GAN.generator_optimizer.iterations, chi))
                        if Pchi < bestPchi:
                            GAN.checkpoint.write(file_prefix = os.path.join(GAN.checkpoint_dir, 'BESTPCHI'))
                            bestPchi = Pchi
                            print('BESTPCHI: iter=%d, Pchi=%f'%(GAN.generator_optimizer.iterations, Pchi))


        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    t_end=time.time()
    print("Total time",t_end-t_begin)
    print('DONE')

