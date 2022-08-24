# 1. split train data and validation set randomly
# 2. open each .fit file that corresponds to the train set
# 3. get flux and wavelength and check for abnormal values
# 4. normalize values
# 5. build the net to detect b/non-b stars
# 6. check with validation set and +- more 400 non b stars
# 7. increase net to other types
# 8. include ~O star type with SAGAN
from os.path import exists
from msilib.schema import Error
import os
import sys
from typing import Tuple, Callable
from astropy.io import fits
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import InputLayer
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from timeit import default_timer as timer
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'extract')
sys.path.append(mymodule_dir)
from data_source import LamostSource, LamostOTypeExtracted
from pre_processing import PreProcessing


# Loads LAMOST DR6 csv metadata
def load_dataset(path: str) -> pd.Series:
    return pd.read_csv(path, sep="|")


# Loads random samples created by the random_sample.py script in the extract folder
def load_randomSamples(path: str) -> np.ndarray:
    with open(path, 'r') as f:
        lines = f.readlines()
        return np.asarray(lines, dtype=np.int32)


# This function shuffles the samples contained in the given training and test set every time is get called
def split_dataset(ser: pd.Series, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    return train_test_split(ser, y, test_size=2940)


# This function buils the CNN utilised during the research project to classify stellar spectra in the LAMOST DR6 II
def evaluate_model(trainX: np.ndarray, trainy: np.ndarray, testX: np.ndarray, testy: np.ndarray):
    verbose, epochs, batch_size = 0, 140, 32
    kernel_s = 3
    model = Sequential()

    model.add(InputLayer(input_shape=(1, 3500)))

    model.add(Conv1D(filters=10, kernel_size=kernel_s,
              activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))

    model.add(Conv1D(filters=20, kernel_size=kernel_s,
              activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))

    model.add(Conv1D(filters=40, kernel_size=kernel_s,
              activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))

    model.add(Conv1D(filters=50, kernel_size=kernel_s,
              activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(11500, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam", metrics=['accuracy'])

    # Train the CNN netowrk
    history = model.fit(trainX, trainy, epochs=epochs,
                        batch_size=batch_size, verbose=verbose,
                        validation_data=(testX, testy))

    # predict answers for the test set
    y = np.argmax(model.predict(testX), axis=1)
    print(y[0])
    print(testy[0])

    # print the confusion matrix
    confusion = confusion_matrix(
        np.argmax(testy, axis=1), y,  normalize='pred')
    print(confusion)

    # plots accuracy and loss graphs
    history_dict = history.history
    print(history_dict.keys())
    _, accuracy = model.evaluate(
        testX, testy, batch_size=batch_size, verbose=0)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Training iteration (epoch)')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Training iteration (epoch)')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return accuracy


# benchmark function to measure how long every function given as an argument takes to run
def benchmark(text: str, *funcs: Callable):
    start = timer()
    results = []
    print(text)
    for i, func in enumerate(funcs):
        results.append(func())
    end = timer()
    print(end - start)
    return results


# trains the same CNN network multiple times, shuffling the data in a different order every time it runs to ensure the model is not overfitted
def run_experiment(X, Y, repeats=5):
    scores = list()
    for r in range(repeats):
        X_train, X_test, y_train, y_test = benchmark(
            "Splitting dataset", lambda: split_dataset(X, Y))[0]

        train_flux = np.asarray([row[0] for i, row in X_train.iterrows()])
        test_flux = np.asarray([row[0] for i, row in X_test.iterrows()])
        print(test_flux.shape)

        accuracy = evaluate_model(train_flux[:, np.newaxis, :], to_categorical(
            y_train), test_flux[:, np.newaxis, :], to_categorical(y_test))
        score = accuracy * 100.0
        scores.append(score)
    # summarize results
    summarize_results(scores)


# print the average and std. deviation obtained as the run_experiment ran
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# The aim of this script is to train a GAN model to produce synthetic spectra.
# NOTE: It is recommended to have run all previous steps before trying to train this CNN model
if __name__ == '__main__':
    print("Starting trainnng")
    X_hasbeen_processed = exists("X_sim.pkl")
    y_hasbeen_processed = exists("y_sim_values.npy")

    # once the data has been preparated and pre-processed, it is wasteful to reprocess it again
    # since the data will always be the same
    if X_hasbeen_processed and y_hasbeen_processed:
        print("Preprocessed files have been found")
        X = pd.read_pickle("X_sim.pkl")
        y = np.load('y_sim_values.npy')
    else:
        print("Preparing data")
        bSample, aSample,  fSample, gSample, kSample, mSample = benchmark("Loading text file",
                                                                          lambda: load_randomSamples(
                                                                              '../dataset/B-sample.txt'),
                                                                          lambda: load_randomSamples(
                                                                              '../dataset/A-sample.txt'),
                                                                          lambda: load_randomSamples(
                                                                              '../dataset/F-sample.txt'),
                                                                          lambda: load_randomSamples(
                                                                              '../dataset/G-sample.txt'),
                                                                          lambda: load_randomSamples(
                                                                              '../dataset/K-sample.txt'),
                                                                          lambda: load_randomSamples(
                                                                              '../dataset/M-sample.txt'))

        dataset, = benchmark("loading LAMOST B csv",
                             lambda: load_dataset('../dataset/B.csv'))
        fSamples, = benchmark("Finding selected B stars in LAMOST B csv",
                              lambda: dataset.loc[(dataset["obsid"].isin(bSample))])

        dataset, = benchmark("loading LAMOST A csv",
                             lambda: load_dataset('../dataset/A.csv'))

        fSamples, = benchmark("Finding selected A stars in LAMOST A csv",
                              lambda: pd.concat([fSamples, dataset.loc[(dataset["obsid"].isin(aSample))]], ignore_index=True))

        dataset, = benchmark("loading LAMOST F csv",
                             lambda: load_dataset('../dataset/F.csv'))

        fSamples, = benchmark("Finding selected F stars in LAMOST F csv",
                              lambda: pd.concat([fSamples, dataset.loc[(dataset["obsid"].isin(fSample))]], ignore_index=True))

        dataset, = benchmark("loading LAMOST G csv",
                             lambda: load_dataset('../dataset/G.csv'))

        fSamples, = benchmark("Finding selected G stars in LAMOST G csv",
                              lambda: pd.concat([fSamples, dataset.loc[(dataset["obsid"].isin(gSample))]], ignore_index=True))

        dataset, = benchmark("loading LAMOST K csv",
                             lambda: load_dataset('../dataset/K.csv'))

        fSamples, = benchmark("Finding selected K stars in LAMOST K csv",
                              lambda: pd.concat([fSamples, dataset.loc[(dataset["obsid"].isin(kSample))]], ignore_index=True))

        dataset, = benchmark("loading LAMOST M csv",
                             lambda: load_dataset('../dataset/M.csv'))

        fSamples, = benchmark("Finding selected M stars in LAMOST M csv",
                              lambda: pd.concat([fSamples, dataset.loc[(dataset["obsid"].isin(mSample))]], ignore_index=True))

        # This function sets the Y values for every category the CNN model will evaluate
        def conditions(x: int):
            if x.startswith("B"):
                return 0
            elif x.startswith("A"):
                return 1
            elif x.startswith("F"):
                return 2
            elif x.startswith("G"):
                return 3
            elif x.startswith("K"):
                return 4
            elif x.startswith("sd") or x.startswith("d") or x.startswith("g"):
                return 5
            else:
                raise ValueError('A very specific bad thing happened.')

        func = np.vectorize(conditions)
        y = np.concatenate((func(fSamples["subclass"]), np.ones(
            1400, dtype=np.int32) * 6), dtype=np.int32).reshape(9800, 1)

        samples = LamostSource(fSamples)
        o_real = LamostOTypeExtracted('../dataset/processed_O-final.csv')
        o_fake = LamostOTypeExtracted(
            '../dataset/processed_artificial-final.csv')

        p = PreProcessing(samples, o_real, o_fake)
        benchmark("pre-processing", lambda: p.process())

        X = pd.concat([pd.DataFrame(p.processedSpectra[0]), pd.DataFrame(
            p.processedSpectra[1]), pd.DataFrame(p.processedSpectra[2])], ignore_index=True)

        # save the files so next time it runs, the pre-processing doesn't get wastefully reapplied
        print("Saving preprocessed data")
        np.save('y_sim_values.npy', y)
        X.to_pickle("X_sim.pkl")

    print("Training")
    run_experiment(X, y)
    print("Done!")
