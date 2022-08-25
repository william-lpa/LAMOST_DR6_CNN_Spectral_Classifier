# LAMOST_DR6_CNN_Spectral_Classifier
This research project aims to classify stellar spectra present in the LAMOST DR6 II data release.

### 1. Project discription
With technological advancements and the tremendous growth of available data in recent years, more large-scale astronomical spectral surveys are being constantly released. This increase leads to millions of celestial bodies' spectra not being adequately analysed by specialists, or often being misclassified. This research examines spectra of stars observed in the second version of the sixth data release of the Large Sky Area Multi-Object Fibre Spectroscopic Telescope (LAMOST DR6 II). A program was written to train a one-dimensional convolutional neural network (1D CNN) to analyse and classify stellar spectra correctly. O-type stars have a shortened life span and are often interpreted as “no signal spectra” due to their weak features. This characteristic makes a released dataset suffer from the imbalanced problem in machine learning, an issue where there is an unequal distribution of classes in the dataset. To counterbalance the imbalance, a spectral Generative Adversarial Network (GAN) was trained to produce synthetic spectra for the O-type stars. The model achieved an overall accuracy of 83.95%, whereas for O-type stars, an accuracy of 99% was obtained, meaning that the GAN network could map an adequate number of unique features from real O-type spectra. The CNN architecture and hyperparameters used in this research, such as the number of feature-maps, learning rate and kernel size, are also evaluated and available in this report.

### Observation
If you want to see the full report or identify any error with this code, don't hesitate to get in touch!

### 2.  How to run the program
- **Development Environment**:
This program was written on a Windows system without any other configuration. The source code is written in Python 3.9
- **Dependencies needed to run the code:**:
    - numpy
    - notebook
    - pandas
    - astropy
    - tensorflow2
    - keras
    - sklearn
    - matplotlib

It is important to notice that these packages were installed using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). MiniConda is an open-source, cross-platform, language-agnostic package manager and environment management system. It was originally developed to solve difficult package management challenges faced by Python.

#### **Run the software**
 This program runs in scripts at different stages.  
 
1. The first thing one wants to do to run the application is to download the CSV file metadata available on the [LAMOST website](http://dr6.lamost.org/v2/catalogue). It is a considerable CSV file containing more than 9 million rows, so it can take some time to download it. 

2. Once the metadata file is downloaded and the python environment is set with all the dependencies, the script named `extract/extract.py` can be run as the following:
```console
~/LAMOST_DR6_CNN_Spectral_Classifier$ python3 extractors/extract.py
```
NB: Before running the script, make sure you update all the variables to point to valid directories and existing files

That command will take the huge dataset and break it down into smaller ones, grouped by the stellar type present on the subclass column in the CSV file, as demonstrated below:
![My Remote Image](https://drive.google.com/uc?export=view&id=1cID_hiVNDitZoIlPORVrBIgIs0KJqiIo)

3. Once the data is organised in different datasets, the script named `extract/extract.py` can be run as the following:
```console
~/LAMOST_DR6_CNN_Spectral_Classifier$ python3 extractors/random_sample.py
```
NB: Before running the script, make sure you update all the variables to point to valid directories and existing files

This step will now get every single one of the newly generated subsets and save random samples to text files containing the selected observation ids on every line. The reason why these ids are randomly extracted is because there is a [search page](http://dr6.lamost.org/v2/catalogue) on the LAMOST website, which allows us to search spectra by a list of observation IDs as illustrated below:
![My Remote Image](https://drive.google.com/uc?export=view&id=1tp6IJYqkUZCkYaMzL-pIKxGVbl72iqKp)

That page can also be used to download the spectra, which are stored in FITS files. These FITS files will be used to train the GAN and CNN classifier also present in this repository.

4. As already stated, the dataset released by LAMOST is imbalanced. So we need to use a GAN network to synthesise additional O-type spectra and fix the imbalance problem. The GAN network needs real O-type spectra from the LAMOST DR6 II database and also from [JACOBY's et al. ](https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/jacoby-hunter-christian-atlas), so it can learn how to map new features and produce additional synthetic spectra of the O-type start.

```console
~/LAMOST_DR6_CNN_Spectral_Classifier$ python3 extractors/extra-O-type.py
```

This script preprocesses spectra from these two sources, filtering and normalising the data. Finally, it saves them into a CSV file where the GAN network can use these processed data to learn how to create artificial spectra.

NB: Before running the script, make sure you update all the variables to point to valid directories and existing files

5. Training GAN
```console
~/LAMOST_DR6_CNN_Spectral_Classifier$ python3 simulation/sgan.py
```
This step can take several hours, depending on your hardware. If you have to stop training for whatever reason, it is possible to continue from where you stopped because of the usage of checkpoints in this code. In order to use checkpoint, you can just rerun the same command. Checkpoints are saved every 50 iterations, and images of the synthetic spectra are displayed every 500 epochs.

6. When GAN has trained for several epochs, it means it will have successfully learnt the necessary features to produce more synthetic O-type spectra. To generate N numbers of spectra and save it on disk, just run the following command: 
```console
~/LAMOST_DR6_CNN_Spectral_Classifier$ python3 simulation/prediction.py
```
NB: Before running the script, make sure you update all the variables to point to valid directories and existing files

7. With a now balanced dataset, we can train the CNN model to classify stars by inspecting their spectra. To train the CNN, just run the following command:
```console
~/LAMOST_DR6_CNN_Spectral_Classifier$ python3 simulation/simulation.py
```
This network can also take some minutes to train, depending on the hardware. This file caches pre-computed values, so at least if this code is run multiple times, the waste of preprocessing the same data multiple times is not going to happen.

Plots will show the accuracy of the training set and test set every time the CNN is trained.

NB: Before running the script, make sure you update all the variables to point to valid directories and existing files

8. The overall results of this research project were very satisfactory. The network achieved an accuracy of 83.95% for all the seven stellar types, whereas it got a remarkable performance of 99% when classifying O-type spectra, meaning the GAN network managed to learn real features from that domain with great fidelity.

Some of the graphics published on the report for the accuracy and loss function can be checked here:
![My Remote Image](https://drive.google.com/uc?export=view&id=1vtfZ1U1KzvguGHLG905se_gGC8WW9m7J)

The confusion matrix is also available to see how well the CNN model performed for each one of the classes:
![My Remote Image](https://drive.google.com/uc?export=view&id=1OF-VlM554KgcsZA8cSM-_Sly6e-0C1PT)

Finally, here are some useful scripts if you want to explore the FITS file present in Jacoby's et al. spectral library and LAMOST's DR6 II database:

```python
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def MaxMinNormalization(x):
    Max = np.max(x)
    Min = np.min(x)
    x = (x - Min) / (Max - Min)
    return x

#load JACOBY file
with fits.open('D:/project/dataset/fits/jc_7.fits') as hdul:
    data = hdul[1].data
    flux = pd.Series(data["FLUX"].byteswap().newbyteorder())
    flux = (flux-flux.min())/(flux.max()-flux.min())
    rolling = flux.rolling(window=3)
    rolling_mean = rolling.mean()
    wl = pd.Series(data["WAVELENGTH"].byteswap().newbyteorder(), name='wl')
    df = pd.concat([wl, flux, rolling_mean], axis=1)
    fDf = df.loc[(df['wl'] >= 3700.0) & (df['wl'] <= 8671.6)]    

upper_limit = 3500 - fDf.shape[0]

normalized_wv = np.pad(fDf['wl'], (0, upper_limit),
                       'constant', constant_values=(0, 0))

normalized_flx = np.pad(fDf[1], (0, upper_limit),
                        'constant', constant_values=(0, 0))

normalized_flx_raw = np.pad(fDf[0], (0, upper_limit),
                        'constant', constant_values=(0, 0))

wv = np.linspace(3700.0, 8671.6, num=3500)

#load LAMOST DR6 II file
with fits.open('D:/project/dataset/fits/spec-56338-GAC114N32B1_sp12-018.fits.gz') as _hdul:
    _hdul.info()  # assuming the first extension is a table
    _data = _hdul[0].data
    _flux = pd.Series(_data[0, :].byteswap().newbyteorder())
    _flux = (_flux-_flux.min())/(_flux.max()-_flux.min())
    _rolling = _flux.rolling(window=3)
    _rolling_mean = _rolling.mean()        
    _wl = pd.Series(_data[2, :].byteswap().newbyteorder(), name='wl')

_df = pd.concat([_wl, _flux, _rolling_mean], axis=1)
_fDf = _df.loc[(_df['wl'] >= 3700.0) & (_df['wl'] <= 8671.6)]



plt.step(wv, _fDf[1][0:3500], where='mid', linewidth=0.5, color='b')
plt.step(wv, normalized_flx, where='mid', linewidth=0.5, color='r')
plt.legend(['LAMOST O-type star', 'JACOBY O-type star'])
plt.xlabel("Wavelength (Angstroms)")
plt.ylabel("Flux")
plt.show()
```
