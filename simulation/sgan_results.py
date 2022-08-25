import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from astropy.io import fits
import pandas as pd
from sgan import make_generator_model


noise = tf.random.normal([1, 900])
generator = make_generator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 8000
noise_dim = 900
num_examples_to_generate = 5
BUFFER_SIZE = 60000
BATCH_SIZE = 3


checkpoint_dir = './training_checkpoints-test-loss'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 generator=generator,
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer
                                 )
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)


seed = tf.random.normal([num_examples_to_generate, noise_dim])


def processFakeSpectra():
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("not found")
        return

    generate_and_save_images(generator, seed)


def generate_and_save_images(model, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    print(predictions.shape)

    wv = np.linspace(3700.0, 8671.6, num=3500)

    fig = plt.figure(figsize=(4, 4))
    fig.suptitle('Artificial Spectra generated by GAN')
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex=True, sharey=True)
    # fig.suptitle('Sharing both axes')
    ax1.step(wv, predictions[0], 'tab:orange')
    ax2.step(wv, predictions[1], 'tab:blue')
    ax3.step(wv, predictions[2], 'tab:green')
    ax4.step(wv, predictions[3], 'tab:red')

    for ax in [ax2, ax3]:
        ax.set(xlabel='Wavelength (Angstroms)', ylabel='Flux')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in [ax1, ax2, ax3, ax4]:
        ax.label_outer()


def compareASpectraVsReal():
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("not found")
        return

    seed = tf.random.normal([1, noise_dim])
    predictions = generator(seed, training=False)        

    with fits.open('D:/project/dataset/fits/spec-58257-GACII259N10M1_sp05-033.fits.gz') as _hdul:
        _hdul.info()  # assuming the first extension is a table
        _data = _hdul[0].data
        _flux = pd.Series(_data[0, :].byteswap().newbyteorder())
        _flux = (_flux-_flux.min())/(_flux.max()-_flux.min())
        _rolling = _flux.rolling(window=3)
        _rolling_mean = _rolling.mean()
        _wl = pd.Series(_data[2, :].byteswap().newbyteorder(), name='wl')
        _df = pd.concat([_wl, _flux, _rolling_mean], axis=1)
        _fDf = _df.loc[(_df['wl'] >= 3700.0) & (_df['wl'] <= 8671.6)]

    fig = plt.figure(figsize=(4, 4))
    fig.suptitle('Comparison of fake vs real spectra')
    wv = np.linspace(3700.0, 8671.6, num=3500)
    plt.step(wv, _fDf[1][0:3500], where='mid', linewidth=0.3, color='tab:blue')
    plt.step(wv, predictions[0], where='mid', linewidth=0.3, color='tab:red')
    plt.xlabel("Wavelength (Angstroms)")
    plt.ylabel("Flux")
    plt.legend(['LAMOST O-type star', 'GAN generated O-type spectrum'])


def plotLossFunction():
    loss_beforeRelu = np.loadtxt("d_loss_O-no-jacoby.csv")
    loss_noRelu = np.loadtxt("d_loss_O-no-jacoby-no-bnorm.csv")
    loss_afterRelu = np.loadtxt("d_loss_O-no-jacoby-bnorm-after-relu.csv")
    
    
    x = np.linspace(0, loss_beforeRelu.shape[0], loss_beforeRelu.shape[0])

    fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 8))
    fig.suptitle('Loss function')    
    axs[0].plot(x, loss_beforeRelu, 'tab:red')
    axs[0].set_title('Batch normalisation before activation function')
    axs[0].set_xlabel('Number of training iterations')
    axs[0].set_ylabel('Value of the loss function')

    axs[1].plot(x, loss_noRelu, 'tab:orange')
    axs[1].set_title('No batch normalisation')
    axs[1].set_xlabel('Number of training iterations')
    axs[1].set_ylabel('Value of the loss function')

    axs[2].plot(x, loss_afterRelu, 'tab:blue')
    axs[2].set_title('Batch normalisation after activation function')
    axs[2].set_xlabel('Number of training iterations')
    axs[2].set_ylabel('Value of the loss function')

    # fig.subplots_adjust(hspace=0.5)
    
    # #second plot
    loss = np.loadtxt("d_loss_O-no-jacoby-final")
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Convergence of Loss Function\nBatch normalisation before activation function')        
    plt.xlabel("Number of training iterations")
    plt.ylabel("Value of the loss function")
    print("AA", loss.shape)
    x = np.linspace(0, loss.shape[0], loss.shape[0])
    plt.step(x,loss,'tab:green')    
    

# visualise results obtained for GAN
if __name__ == '__main__':
    plotLossFunction()
    processFakeSpectra()
    compareASpectraVsReal()