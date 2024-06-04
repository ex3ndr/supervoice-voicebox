import torch
import torchaudio
import multiprocessing
from pathlib import Path
from tqdm import tqdm

def main():

    # Load datasets
    datasets = []
    for d in Path("datasets").glob("*-aligned"):
        datasets.append(d.stem.removesuffix('-aligned'))

    for dataset in datasets:

        # Enumerate files        
        print("Loading files from " + dataset + " dataset...")
        files = []
        for f in Path("datasets").joinpath(dataset + "-aligned").glob("*/*.TextGrid"):
            files.append(str(f).removesuffix('.TextGrid').removeprefix('datasets/' + dataset + '-aligned/'))
        files.sort()
        
        # Write index
        print("Writing index for " + dataset + " dataset...")
        with open("datasets/" + dataset + ".txt", "w") as index:
            for file in files:
                index.write(file + "\n")
                

if __name__ == "__main__":
    main()