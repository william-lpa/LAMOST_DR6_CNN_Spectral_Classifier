import sys
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# add this to the modules, so pyrhon can find the extract folder and interpret it as a module
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'extract')
sys.path.append(mymodule_dir)

from tensorflow.keras import layers
from data_source import JacobySource, LamostSource

# Generator Model
def make_generator_model():
    generator = tf.keras.Sequential()
    generator.add(layers.Dense(256, input_dim=noise.shape[1],
                  kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
    generator.add(layers.LeakyReLU(0.2))
    generator.add(layers.Dense(512))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU(0.2))

    generator.add(layers.Dense(1024))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU(0.2))
    generator.add(layers.Dense(3500, activation='tanh'))

    return generator

# Discriminator Model


def make_discriminator_model():
    discriminator = tf.keras.Sequential()
    discriminator.add(layers.Dense(1024, input_dim=3500,
                      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
    discriminator.add(layers.LeakyReLU(0.2))
    discriminator.add(layers.Dropout(0.3))

    discriminator.add(layers.Dense(512))
    discriminator.add(layers.LeakyReLU(0.2))
    discriminator.add(layers.Dropout(0.3))

    discriminator.add(layers.Dense(256))
    discriminator.add(layers.LeakyReLU(0.2))
    discriminator.add(layers.Dropout(0.3))

    discriminator.add(layers.Dense(1, activation='sigmoid'))

    return discriminator


# Random-noise vector
noise = tf.random.normal([1, 900])

# create the generator network
generator = make_generator_model()

# create the discriminator network
discriminator = make_discriminator_model()


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# calculate discrimintor loss loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# calculate generator loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Learning Optimisers utilised for the Generator and Discriminator network
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# how long the algorithm will train
EPOCHS = 8000
# noise vector-length
noise_dim = 900

BUFFER_SIZE = 60000
BATCH_SIZE = 3

# The idea here is to periodically save the model since this network takes a long time to converge

# Where the save the file that will persist the last state
checkpoint_dir = './training_checkpoints-test-loss-no-jacoby-final'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# The idea here is to periodically save the model since this network takes a long time to converge
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)


def load_real_spectra():
    # this function loads the spectra that will be used as template to compare the similarity
    #  of the artificially generated spectra
    spectra = []
    jacoby = JacobySource("NOOP")
    lamost = LamostSource("NOOP")

    spectra.append(jacoby.normalise(jacoby.filter(jacoby.window_avg(
        jacoby.load_file("D:/project/dataset/fits/jc_7.fits")))))
    spectra.append(lamost.normalise(lamost.filter(lamost.window_avg(lamost.load_file(
        "D:/project/dataset/fits/spec-58257-GACII259N10M1_sp05-033.fits.gz")))))
    spectra.append(lamost.normalise(lamost.filter(lamost.window_avg(lamost.load_file(
        "D:/project/dataset/fits/spec-56338-GAC114N32B1_sp12-018.fits.gz")))))

    return spectra


# Scale normalisation function
def MaxMinNormalization(x):
    Max = np.max(x)
    Min = np.min(x)
    x = (x - Min) / (Max - Min)
    return x


# Takes the Euclidean distance of the generated spectra vs the real ones
# to se how close they are. That is used a objective cost function for the
# model
def similarity_O(generated, realData):
    similarity = []
    f = generated.shape[0]
    for j in range(f):
        sampleSimilarities = []
        for sample in realData:
            generated[j] = MaxMinNormalization(generated[j])
            c = generated[j]-sample
            a = c/sample
            where_are_nan = np.isnan(a)
            where_are_inf = np.isinf(a)
            a[where_are_nan] = 0
            a[where_are_inf] = 0
            similarity_n = np.std(a, ddof=1)
            sampleSimilarities.append(similarity_n)
        similarity.append(min(sampleSimilarities))
    similarity_mean = np.mean(similarity)
    return similarity_mean


# number of spectra that will be printed every 500 steps during training
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# this is the function where the generator and discriminator are trained in a competitive
# way the generator gets trained first. Later fake and real spectra are mixed up and submitted
# to the discriminator, which will also be trained according to its performance.
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        fake_output = discriminator(generated_images, training=True)
        real_output = discriminator(images[np.newaxis, :], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


# this function manages the training process. It calls the train_step function on every epoch,
# it saves the checkpoint every 50 iterations and it also displays how the current synthetic spectra
# look like at that time of training
def train(dataset, epochs, similarityData):
    # Here we need to check if there is a checkpoint. If there is, load the parameters and continue
    # from there. Otherwise, start a new trainning process from scratch.
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    loss = []

    for epoch in range(epochs):

        checkpoint.step.assign_add(1)
        # print current some spectra to see how they are looking
        if (epoch) % 500 == 0:
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)

        # often save the current state, so we don't loose current progress
        if (epoch) % 50 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(
                int(checkpoint.step), save_path))
            with open("d_loss_O.csv", "ab") as f:
                lossNP = np.array(loss)
                np.savetxt(f, lossNP, delimiter=',')
                loss.clear()

        for image_batch in dataset:
            train_step(image_batch)
        noise = np.random.normal(0, 1, size=[177, 900])
        generated_images = generator.predict(noise, verbose=0)
        loss.append(similarity_O(generated_images, similarityData))

    generate_and_save_images(generator,
                             epochs,
                             seed)


# generate a linear space for the wavelengths
wv = np.linspace(3700.0, 8671.6, num=3500)


# print some spectra to see how they look like
def generate_and_save_images(model, epoch, test_input):

    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    print(test_input.shape)
    predictions = model(test_input, training=False)
    print("here", predictions.shape)

    fig = plt.figure(figsize=(4, 4))

    for i in range(8, predictions.shape[0]):
        #plt.subplot(4, 4, i+1)
        plt.step(wv, predictions[i])
    plt.show()

# loads the preprocesed csv file containing data from LAMOST and JACOBY distribution


def load_data():
    x_train = pd.read_csv('../dataset/processed_O.csv',
                          header=None, dtype=np.float32)
    x_train = np.array(x_train)[19:]
    print(x_train[0][0])
    return (x_train)


# The aim of this script is to train a GAN model to produce synthetic spectra.
# NOTE: It is recommended to first run the extract_O-type.py script in the extract folder to preprocess O-type spectra from the LAMOST DR6 II and JAcoby's et al. spectral library
if __name__ == '__main__':
    real_spectra = load_real_spectra()
    train(load_data(), EPOCHS, [a[0] for a in real_spectra])
