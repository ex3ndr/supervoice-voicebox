import csv
import os
import multiprocessing
import glob
import pandas as pd
from tqdm import tqdm
from utils.dataset import load_common_voice_files
import torchaudio

#
# Parameters
#

PARAM_COMMON_VOICE_PATH = "external_datasets/common-voice-*/*/"
PARAM_WORKERS = multiprocessing.cpu_count()

#
# Execution
#

def execute_parallel_verify(args):
    files, index = args
    try:
        torchaudio.load(files[index])
        return files[index]
    except:
        print(f"Invalid file: {files[index]}")
        return None

def execute_parallel(args):
    files, index = args
    return load_common_voice_files(files[index], "test"), load_common_voice_files(files[index], "train")

def execute_run():

    # Indexing files
    print("Loading languages...")
    languages = glob.glob(PARAM_COMMON_VOICE_PATH)

    # Indexes loop
    print("Loading indexes...")
    output_test = []
    output_train = []
    with multiprocessing.Manager() as manager:
        files = manager.list(languages)
        args_list = [(files, i) for i in range(len(files))]
        with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
            for result in tqdm(pool.imap_unordered(execute_parallel, args_list), total=len(files)):
                output_test = output_test + result[0]
                output_train = output_train + result[1]
    
    # Indexes loop
    print("Test File verification...")
    output_test_ok = []
    output_test_ok_count = 0

    with multiprocessing.Manager() as manager:
        files = manager.list(output_test)
        args_list = [(files, i) for i in range(len(files))]
        with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
            for result in tqdm(pool.imap_unordered(execute_parallel_verify, args_list, chunksize=2048), total=len(files)):
                if result is not None:
                    output_test_ok.append(result)
    print("Train File verification...")
    output_train_ok = []
    with multiprocessing.Manager() as manager:
        files = manager.list(output_train)
        args_list = [(files, i) for i in range(len(files))]
        with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
            for result in tqdm(pool.imap_unordered(execute_parallel_verify, args_list, chunksize=2048), total=len(files)):
                if result is not None:
                    output_train_ok.append(result)

    # Save results
    print("Saving results...")
    output_test = pd.DataFrame(output_test_ok)
    output_test.to_pickle("./datasets/cv_validated_test.pkl")
    output_train = pd.DataFrame(output_train_ok)
    output_train.to_pickle("./datasets/cv_validated_train.pkl")

if __name__ == "__main__":
    execute_run()