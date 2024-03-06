import torch
import torchaudio
import multiprocessing
from glob import glob
from tqdm import tqdm
from supervoice.model_style import _convert_to_continuous_f0

def get_stats(path):
    base_path = path[:-4]
    style = torch.load(base_path + ".style.pt")
    style = _convert_to_continuous_f0(style)
    style = (style - style.mean()) / style.std()
    return (style.mean().item(), style.std().item(), style.max().item(), style.min().item())

def main():

    # Enumerate files
    print("Enumerating files...")
    # files = []
    # files += glob("datasets/libritts-prepared/*/*.wav")
    # files += glob("datasets/vctk-prepared/*/*.wav")
    files_eval = []
    files_eval += glob("datasets/eval-prepared/*/*.wav")
    ops = [("list_test", files_eval)]

    # Calculate duration
    for op in ops:
        print(f"Calculating stats for {op[0]}")

        stats = []
        with multiprocessing.Manager() as manager:
            files = manager.list(op[1])
            with multiprocessing.Pool(processes=32) as pool:
                for result in tqdm(pool.imap_unordered(get_stats, files, chunksize=32), total=len(files)):
                    stats.append(result)
        
        # Extract max and min
        max_mean = max([x[0] for x in stats])
        min_mean = min([x[0] for x in stats])
        max_std = max([x[1] for x in stats])
        min_std = min([x[1] for x in stats])
        max_max = max([x[2] for x in stats])
        min_max = min([x[2] for x in stats])
        max_min = max([x[3] for x in stats])
        min_min = min([x[3] for x in stats])

        # Log
        print("Mean: ", max_mean, " - ", min_mean)
        print("Std: ", max_std, " - ", min_std)
        print("Max: ", max_max, " - ", min_max)
        print("Min: ", max_min, " - ", min_min)
                

if __name__ == "__main__":
    main()