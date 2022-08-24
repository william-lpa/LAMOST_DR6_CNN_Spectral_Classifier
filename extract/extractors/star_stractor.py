import pandas as pd
import os


class StarExtractor:
    """ A class responsible for extracting the provided class and subclass in LAMOST metadata.

    Attributes:
        source: stores the the location of where the LAMOST DR6 II metadata is
        _class: stores the class column in the LAMOST DR6 II metadata
        _subclass: stores the sublass column in the LAMOST DR6 II metadata
    """

    def __init__(self, source: str, _class: str, _subclass: str):
        """ Initialize the class with the given parameters.

        Args:
            source: The location of where the LAMOST DR6 II metadata is
            _class: That refers to the class column in the LAMOST DR6 II metadata
            _subclass: That refers to the sublass column in the LAMOST DR6 II metadata
        """
        self.source = source
        self._class = _class
        self._subclass = _subclass

    def saveCSV(self, path: str) -> None:
        """ Saves the generated subset extracted from the LAMOST DR6 and save it on the given path.

        Args:
            path: path where the extracted subset will be stored

        """
        self.data.to_csv(os.path.join(
            path, self._subclass + '.csv'), sep="|", index=None)

    def getObjectType(self) -> pd.DataFrame:
        """ Applies an exact match on the stellar class and subclass received in the StarExtractor constructor and returns it as a Pandas DataFrame.

        Returns:
            pd.DataFrame containing all found intances for the provided stellar type

        """
        metadata = pd.read_csv(self.source, sep="|", low_memory=False)
        return metadata.loc[(metadata["class"] == self._class) & (metadata["subclass"].str.startswith(self._subclass))]

    def extract(self, extractToPath: str):
        self.data = self.getObjectType()
        self.saveCSV(extractToPath)


class MStarExtractor(StarExtractor):
    """ This class inherits from base StarExtractor class to produce a specialised M-type start extractor.
        The only function it overrides is the getObjectType since M-type stars in the LAMOST DR6 II survey
        were assign to multiple different subclass

    Attributes:
        source: stores the the location of where the LAMOST DR6 II metadata is
        _class: stores the class column in the LAMOST DR6 II metadata
        _subclass: stores the sublass column in the LAMOST DR6 II metadata

    """

    def getObjectType(self) -> pd.DataFrame:
        """ Finds all M-type stars stored in the LAMOST DR6 II metadata. LAMOST has released M-type stars with the following prefixes:
        'sd' = 'subdwarf'
        'g' = 'Giant'
        'g' = 'dwarf'

        All these subclasses will be considered in order to produce the subset data

        Returns:
            pd.DataFrame containing all found intances for the M-type class

        """
        metadata = pd.read_csv(self.source, sep="|", low_memory=False)
        return metadata.loc[(metadata["class"] == self._class) & ((metadata["subclass"].str.startswith("sd"+self._subclass)) |
                                                                  (metadata["subclass"].str.startswith("d"+self._subclass)) |
                                                                  (metadata["subclass"].str.startswith("g"+self._subclass)))]


class OStarExtractor(StarExtractor):
    """ This class inherits from base StarExtractor class to produce a specialised pure O-type start extractor.
         The only function it overrides is the getObjectType since O-type stars in the LAMOST DR6 II survey
         were the O and OB subcategories

        Attributes:
            source: stores the the location of where the LAMOST DR6 II metadata is
            _class: stores the class column in the LAMOST DR6 II metadata
            _subclass: stores the sublass column in the LAMOST DR6 II metadata

    """

    def getObjectType(self) -> pd.DataFrame:
        """ Finds all O-type stars stored in the LAMOST DR6 II metadata by doing an exact match using the _subclass class attribute            

            Returns:
                pd.DataFrame containing all found intances for the O-type class

            """
        metadata = pd.read_csv(self.source, sep="|", low_memory=False)
        return metadata.loc[(metadata["class"] == self._class) & (metadata["subclass"] == self._subclass)]
