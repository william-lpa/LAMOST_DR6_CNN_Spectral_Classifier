from operator import mod
import os
import pandas as pd

# The aim of this script is to randomly select N instances from the previously extracted dataset.
# NOTE: It is recommended to first run the extract.py script to separate the mixed stars classes into multiple smaller datasets

if __name__ == '__main__':
    # how many random samples will be extracted to the text file
    number_of_random_spectra_to_extract = 1400

    # the value of what column from the metadata file will be stored in the text file
    exported_column = "obsid"

    # change this to the directory where the previously generated subsets are
    # e.g. "D:\project\dataset"
    source_dir = "D:\project\dataset"

    def sampleSubsetData(path: str, number: int) -> pd.Series:
        train = pd.read_csv(path, sep="|")
        return train.sample(n=number)

    for stellar_type in ["O", "B", "A", "F", "G", "K", "M"]:
        print(f"Ramdonly sampling {number_of_random_spectra_to_extract} {stellar_type}-type stars")
        if stellar_type == "O":
          # reading subset metadata
            serie: pd.Series = sampleSubsetData(os.path.join(
                source_dir, f"{stellar_type}.csv"), 122)[exported_column]
        else:
          # reading subset metadata
            serie: pd.Series = sampleSubsetData(os.path.join(
                source_dir, f"{stellar_type}.csv"), number_of_random_spectra_to_extract)[exported_column]

        # exporting random ids from the subset metadata
        serie.to_csv(os.path.join(
            source_dir, f"{stellar_type}-sample.txt"), header=None, index=None)
        print("Done!")
