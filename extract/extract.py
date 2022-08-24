import os

from extractors.star_stractor import StarExtractor
from extractors.star_stractor import MStarExtractor
from extractors.star_stractor import OStarExtractor


# The aim of this script is to extract subsets derived from the released LAMOST DR6 II metadata containg information about the observed bodies
if __name__ == '__main__':

    # change this to the LAMOST DR6 II file metadata name
    lamost_dr6_v2_csv = 'LAMOST-DR6-V2-FILE-NAME'  # e.g. dr6_v2_LRS.csv

    # change this to the directory where LAMOST DR6 II file is
    lamost_dr6_v2_csv_dir = 'LAMOST-DR6-V2-FILE-LOCATION'  # e.g. 'D:/project/'

    # the first argument needs to be the directory which the csv metadata file is
    lamost_metadata = os.path.join(lamost_dr6_v2_csv_dir, lamost_dr6_v2_csv)

    # set a suitable destination directory
    extraction_folder = '../dataset'

    # loop through all Mk classification system stars
    for stellar_type in ["O", "B", "A", "F", "G", "K", "M"]:
        print(f"Extracting {stellar_type}-type star")

        extractor = None

        if stellar_type == "O":
            extractor = OStarExtractor(lamost_metadata, "STAR", stellar_type)
        if stellar_type == "M":
            extractor = MStarExtractor(lamost_metadata, "STAR", stellar_type)
        else:
            extractor = StarExtractor(lamost_metadata, "STAR", stellar_type)

       # files will be saved into 'extraction_folder/{stellar_type}.csv directory'
        extractor.extract(extraction_folder)
        
        print("Done!")
