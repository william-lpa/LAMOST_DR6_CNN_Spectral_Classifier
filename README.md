# LAMOST_DR6_CNN_Spectral_Classifier
It is a research that aims to classify stellar spectra present in the LAMOST DR6 II data release

### 1. Project discription
With technological advancements and the tremendous growth of available data in recent years, more large-scale astronomical spectral surveys are being constantly released. This increase leads to millions of celestial bodies' spectra not being adequately analysed by specialists, or often being misclassified. This research examines spectra of stars observed in the second version of the sixth data release of the Large Sky Area Multi-Object Fibre Spectroscopic Telescope (LAMOST DR6 II). A program was written to train a one-dimensional convolutional neural network (1D CNN) to analyse and classify stellar spectra correctly. O-type stars have a shortened life span and are often interpreted as “no signal spectra” due to their weak features. This characteristic makes a released dataset suffer from the imbalanced problem in machine learning, an issue where there is an unequal distribution of classes in the dataset. To counterbalance the imbalance, a spectral Generative Adversarial Network (GAN) was trained to produce synthetic spectra for the O-type stars. The model achieved an overall accuracy of 83.95%, whereas for O-type stars, an accuracy of 99% was obtained, meaning that the GAN network could map an adequate number of unique features from real O-type spectra. The CNN architecture and hyperparameters used in this research, such as the number of feature-maps, learning rate and kernel size, are also evaluated and available in this report

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
 This program runs in scripts for several different stages.  
 
1. The first thing one wants to do to run the application is to download the CSV file metadata available in the [LAMOST website](http://dr6.lamost.org/v2/catalogue). It is a huge CSV file, containing more than 9 million rows, so it can take some time to download it. 

2. Once the metadata file is downloaded and the python enviroment is set with all the dependencies, the script named `extract/extract.py` can be run as the following:
```console
~/LAMOST_DR6_CNN_Spectral_Classifier$ python3 extractors/extract.py
```
NB: Before running the script, make sure you update all the variables to point to valid directories and existing files

That command will take the huge dataset and break it down into smaller ones, grouped by the stellar type set by the subclass column in the CSV file as demonstrated below:
![My Remote Image](https://drive.google.com/uc?export=view&id=1cID_hiVNDitZoIlPORVrBIgIs0KJqiIo)

3. Once the data is organised in different datasets, the script named `extract/extract.py` can be run as the following:
```console
~/LAMOST_DR6_CNN_Spectral_Classifier$ python3 extractors/random_sample.py
```
NB: Before running the script, make sure you update all the variables to point to valid directories and existing files

This step will now get every single one of the subsets and save random samples to text files, containing the observation id on every line. The reason why these ids are randomly extracte is because there is a [search page](http://dr6.lamost.org/v2/catalogue) in lamost website which allows us to search spectra by a list of observation ids as ilustrated below:
![My Remote Image](https://drive.google.com/uc?export=view&id=1tp6IJYqkUZCkYaMzL-pIKxGVbl72iqKp)

That page can also be used to download the spectra which are stored in FITS files. These fits files will be used to train the CNN classifier also present in this repository.

4. The GAN network needs real O-type spectra from the LAMOST DR6 II database and also from [JACOBY's et al. ](https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/jacoby-hunter-christian-atlas) so it can learn how to map new features and produce addditional synthetic spectra.

```console
~/LAMOST_DR6_CNN_Spectral_Classifier$ python3 extractors/extra-O-type.py
```

This script preprocesses spectra from these two sources, filtering and normalising the data. Finally, it saves them into a CSV file where the GAN network can use these processed data to learn how to create artificial spectra.

NB: Before running the script, make sure you update all the variables to point to valid directories and existing files

```console
~/LAMOST_DR6_CNN_Spectral_Classifier$ python3 simulation/sgan.py
```
This step can take several hours depending on your hardware. If you have to stop training for whatever reason, it is possible to continue from where you stoped because of the usage of checkpoints in this code. In order to use checkpoint, you can just run the same command again. Checkpoints are saved every 50 iterations and images of the artificial spectra are displayed every 500 epochs.
