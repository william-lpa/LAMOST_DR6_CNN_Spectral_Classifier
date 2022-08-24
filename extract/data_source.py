import os
from typing import Tuple
from astropy.io import fits
import numpy as np
import pandas as pd


class DataSource:
    """ An absctract class for reading a dataset from a specific source.

    Attributes:
        path: dataset folder.
    """

    def __init__(self, path: str):
        """ Initialize the class taking a a string.
        """
        self.path = path

    def __init__(self, dataFrame: pd.DataFrame):
        """ Initialize the class with taking a Pandas dataframe.
        """
        self.df = dataFrame

    def fetch(self) -> pd.Series:
        """ Fetches the dataset from a given path.

        Raises:
            NotImplementedError: This virtual method should be overridden by subclassing Datasource.
        """
        raise NotImplementedError()

    def load_file(self, fileName: str) -> Tuple[pd.Series, pd.Series]:
        """ load a given file from the specified location.

        Args:
            fileName: path where the file needs to be stored.

        Raises:
            NotImplementedError: This virtual method should be overridden by subclassing Datasource.
        """
        raise NotImplementedError()

    def window_avg(self, spec: Tuple[pd.Series, pd.Series]) -> Tuple[pd.Series, pd.Series]:
        """ applies a 3-points window average in the spectra wavelength.

        Args:
            spec: Tuple[pd.Series, pd.Series] containing the (flux, wavelength).        
        """
        rolling = spec[0].rolling(window=3)
        rolling_mean = rolling.mean()
        return (rolling_mean, spec[1])

    def filter(self, spec: Tuple[pd.Series, pd.Series]) -> Tuple[pd.Series, pd.Series]:
        """ Apply a filter to the spectra range of 3.700-8671.6 Angstroms.

        Args:
            spec: Tuple[pd.Series, pd.Series] containing the (flux, wavelength).       
        """
        df = pd.concat([spec[0], spec[1]], axis=1)
        fDf = df.loc[(df['wl'] >= 3700.0) & (df['wl'] <= 8671.6)]
        return (pd.Series(fDf['flux'][2:3502]), fDf['wl'][2:3502])

    def normalise(self, spec: Tuple[pd.Series, pd.Series]) -> Tuple[pd.Series, pd.Series]:
        """ By default, this class assumes we are applying a standard scaling normalising.

        Args:
            spec: A tuple containing the flux and wavelength.
        """
        flux = spec[0]
        return ((flux-flux.min())/(flux.max()-flux.min()), spec[1])


class JacobySource(DataSource):
    """ Subclass, inherits from DataSource. JacobySource dataset are  composed 19 O-type spectra.

    Attributes:
        path: folder where Jacoby's library is stored.        
    """

    def __init__(self, path: str):
        """ Initialize the class with empty variables.
        """
        self.path = path

    def fetch(self) -> pd.Series:
        """ Implements the fetch function defined by the abstract DataSource class.

        Returns:
            pd.DataFrame containing all O-type stars available in that library.

        """
        return pd.Series(["jc_1.fits", "jc_2.fits", "jc_3.fits", "jc_4.fits", "jc_5.fits",
                          "jc_6.fits", "jc_7.fits", "jc_8.fits", "jc_9.fits", "jc_10.fits",
                          "jc_62.fits", "jc_63.fits", "jc_64.fits", "jc_65.fits", "jc_114.fits",
                          "jc_115.fits", "jc_156.fits", "jc_157.fits", "jc_158.fits",
                          ])

    def filter(self, spec: Tuple[pd.Series, pd.Series]) -> Tuple[pd.Series, pd.Series]:
        """ Apply a filter to the spectra range of 3.700-8671.6 Angstroms. After the filter, we add zero-padding to the vector length is 1x3500

        Args:
            spec: Tuple[pd.Series, pd.Series] containing the (flux, wavelength).       
        """
        # call data source basic filter
        filtered = super().filter(spec)
        # check how many zero-paddings will be needed
        upper_limit = 3500 - filtered[0].shape[0]

        # apply padding n times
        normalized_flx = np.pad(filtered[0].values, (0, upper_limit),
                                'constant', constant_values=(0, 0))
        return (pd.Series(normalized_flx), spec[1])

    def load_file(self, fileName: str) -> Tuple[pd.Series, pd.Series]:
        """ load a given file from the specified location

        Args:
            fileName: path where the FITS file needs to be stored

        Returns:
            Tuple[pd.Series, pd.Series] containing the (flux, wavelength)
        """

        # e.g. 'D:/project/dataset/fits/'
        jacoby_fits_folder = "FOLDER-LOCATION-WHERE-JACOBY-FITS-FILES-ARE"

        with fits.open(os.path.join(jacoby_fits_folder, fileName)) as hdul:
            data = hdul[1].data
        flux = pd.Series(data["FLUX"].byteswap().newbyteorder(), name='flux')
        wl = pd.Series(data["WAVELENGTH"].byteswap().newbyteorder(), name='wl')

        return (flux, wl)


class LamostSource(DataSource):
    """ Subclass, inherits from DataSource. LamostSource dataset fetches all FITS file stored in the given path

    Attributes:
        path: folder where the LAMOST FITS files are.        
    """

    def __init__(self, path: str):
        """ Initialize the class taking a a string.
        """
        self.path = path

    def __init__(self, dataFrame: pd.DataFrame):
        """ Initialize the class with taking a Pandas dataframe.
        """
        self.df = dataFrame
        self.path = None

    def fetch(self) -> pd.Series:
        """ Implements the fetch function defined by the abstract DataSource class.
        The fits files produced by the LAMOST survey have the "spec-{lmjd}_sp{spid}-{fiberid}.fits.gz" format

        Returns:
            pd.DataFrame containing all O-type stars available in that library.

        """
        if self.path != None:
            self.df = pd.read_csv(self.path, sep="|")

        lmjd = self.df["lmjd"].astype("string")
        planId = self.df["planid"].astype("string")
        spid = self.df["spid"].apply(
            lambda x: '{0:0>2}'.format(x)).astype("string")
        fiberId = self.df["fiberid"].apply(
            lambda x: '{0:0>3}'.format(x)).astype("string")
        dash = np.full(len(lmjd), '-')

        return (np.full(len(lmjd), 'spec-') + lmjd
                + dash + planId +
                np.full(len(lmjd), '_sp') + spid
                + dash + fiberId
                + np.full(len(lmjd), '.fits.gz'))

    def load_file(self, fileName: str) -> Tuple[pd.Series, pd.Series]:
        """ load a given LAMOST spectra file from the specified location

        Args:
            fileName: path where the FITS file needs to be stored

        Returns:
            Tuple[pd.Series, pd.Series] containing the (flux, wavelength)
        """

        # e.g. 'D:/project/dataset/fits/'
        lamost_fits_folder = "FOLDER-LOCATION-WHERE-JACOBY-FITS-FILES-ARE"

        with fits.open(os.path.join(lamost_fits_folder, fileName)) as hdul:
            data = hdul[0].data
        flux = pd.Series(data[0, :].byteswap().newbyteorder(), name='flux')
        wl = pd.Series(data[2, :].byteswap().newbyteorder(), name='wl')
        return (flux, wl)


class LamostOTypeExtracted(DataSource):
    """ Subclass, inherits from DataSource. LamostOTypeExtracted class is a facilitator to the articilially generated
    spectra to be equally consumed by the PreProcessing class when the model gets trained

    Attributes:
        path: folder where the LAMOST FITS files are.        
    """

    def __init__(self, path: str):
        """ Initialize the class taking a a string.
        """
        self.path = path

    def filter(self, spec: Tuple[pd.Series, pd.Series]) -> Tuple[pd.Series, pd.Series]:
        # No operation, the fake spectra is already normalised
        return spec

    def fetch(self) -> pd.Series:
        # Fetch artificially generated spectra
        self.df = pd.read_csv(self.path, sep=",", header=None)
        return pd.Series(np.arange(self.df.shape[0]))

    def window_avg(self, spec: Tuple[pd.Series, pd.Series]) -> Tuple[pd.Series, pd.Series]:
        # No operation, the fake spectra is already averaged
        return spec

    def load_file(self, index) -> Tuple[pd.Series, pd.Series]:
        # No operation
        return (self.df.iloc[index], pd.Series(None, name="wl", dtype=np.float64))
