import csv
import os
import multiprocessing
import glob
import pandas as pd
from tqdm import tqdm
from utils.dataset import load_common_voice_files

#
# Parameters
#

PARAM_COMMON_VOICE_PATH = "external_datasets/common-voice-*/*/"
PARAM_WORKERS = multiprocessing.cpu_count()

#
# Execution
#

def execute_parallel(args):
    files, index = args
    return load_common_voice_files(files[index], "test"), load_common_voice_files(files[index], "train")

def execute_run():

    # Indexing files
    print("Loading languages...")
    languages = glob.glob(PARAM_COMMON_VOICE_PATH)

    # Detector loop
    print("Detecting...")
    output_test = []
    output_train = []
    with multiprocessing.Manager() as manager:
        files = manager.list(languages)
        args_list = [(files, i) for i in range(len(files))]
        with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
            for result in tqdm(pool.imap_unordered(execute_parallel, args_list), total=len(files)):
                output_test = output_test + result[0]
                output_train = output_train + result[1]

    # Save results
    print("Saving results...")
    output_test = pd.DataFrame(output_test)
    output_test.to_pickle("./datasets/cv_validated_test.pkl")
    output_train = pd.DataFrame(output_train)
    output_train.to_pickle("./datasets/cv_validated_train.pkl")

if __name__ == "__main__":
    execute_run()