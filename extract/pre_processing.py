from data_source import DataSource, JacobySource, LamostSource
import os
import pandas as pd
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..')
sys.path.append(mymodule_dir)


class PreProcessing:
    """ Pre-process a list of DataSource class.


    Attributes:
        args: list of datasources that will be processed
    """

    def __init__(self, *args: DataSource):
        """ Initialize the class with empty variables
        """
        self.sources = args
        self.processedSpectra = []

    def process(self) -> None:
        """ Process the list of given datasources. Every one of them, will be fetched, taken the window average, filtered the spectra to the same shape and finally normalised
        """
        for src in self.sources:
            self.processedSpectra.append([src.normalise(src.filter(src.window_avg(src.load_file(s))))
                                          for s in src.fetch()])

    def saveCSV(self, path: str, fileName: str) -> None:
        """ Saves the preprocessed spectra to a CSV fie

        Args:
            path: path where the preprocessed csv file will be saved
            fileName: CSV fileName
        
        """
        for src in self.processedSpectra:
            series = [row[0].reset_index(drop=True)
                      for row in src]
            df = pd.DataFrame(series)
            df.to_csv(os.path.join(path, fileName + '.csv'), mode='a', sep=",",
                      index=None, header=False, na_rep='Unknown')



