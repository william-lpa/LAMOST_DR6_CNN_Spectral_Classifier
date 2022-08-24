import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sgan import generator, generator_optimizer, discriminator_optimizer, discriminator
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits


# first we need to retrieve the checkpoint created by a trained GAN network.
# this check point is produced by the sgan.py file
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)


# Normalisation function
def MaxMinNormalization(x):
    Max = np.max(x)
    Min = np.min(x)
    x = (x - Min) / (Max - Min)
    return x


# Plots some of the generated fake spectra
def generate_and_save_images(predictions):
    wv = np.linspace(3700.0, 8671.6, num=3500)
    fig = plt.figure(figsize=(4, 4))

    for i in range(0, 10):
        plt.step(wv, predictions[i])
    plt.show()


# The aim of this script is to produce artificial spectra once a GAN network has been previously trained.

# NOTE: It is recommended to first run the sgan.py script so the checkpoints can produce meaningful spectra
if __name__ == '__main__':
    # read the checkpoint
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    
    # generate as many spectra(1278) as we need here
    noise = np.random.normal(0, 1, size=[1278, 900])
    generated = generator.predict(noise, verbose=0)
    f = generated.shape[0]

    generate_and_save_images(generated)
    df = pd.DataFrame(generated)

    #save the synthetic spectra to a CSV file
    df.to_csv(os.path.join("../dataset", "processed_artificial-final" + '.csv'), mode='a', sep="|",
              index=None, header=False, na_rep='Unknown')
