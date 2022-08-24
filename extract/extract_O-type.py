import os
from pre_processing import JacobySource
from pre_processing import LamostSource
from pre_processing import PreProcessing

# The aim of this script is to combine Jacoby's et al. dataset and the LAMOST DR6 II data into a single csv file to train the GAN network.
# NOTE: It is recommended to first run the extract.py script to separate the mixed stars classes into multiple smaller datasets
if __name__ == '__main__':
  # jacoby class is self-contained, it knows the file list to fetch the o-type fits files
    print("Starting")
    jacoby = JacobySource("NOOP")

    # change this to the directory where the previously generated subsets are
    # e.g. D:/project/dataset
    lamost_O_csv_metadata_dir = "FOLDER-LOCATION-WHERE-O-SUBSET-IS"
    lamost_O_csv_metadata_file = "O-SUBSET-FILE-NAME"  # O.csv

    lamost = LamostSource(os.path.join(
        lamost_O_csv_metadata_dir, lamost_O_csv_metadata_file))
    p = PreProcessing(jacoby, lamost)

    # fetch, load, filter, normalise
    p.process()

    # save csv with pre-processed O-type stars
    p.saveCSV(lamost_O_csv_metadata_dir, "processed_O")
    print("Finished")
