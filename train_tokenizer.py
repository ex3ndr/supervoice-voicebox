from glob import glob
from tqdm import tqdm
import textgrid

# Load files
dataset_dir = "datasets/libritts-aligned"
files = glob(dataset_dir + "/**/*.TextGrid")

# Load textgrids
tg = [textgrid.TextGrid.fromFile(f) for f in tqdm(files)]

# Extract text
existing = {}
tokens = []
for f in tqdm(tg):
    for t in f[1]:
        if t.mark != '' and t.mark not in existing:
            tokens.append(t.mark)
            existing[t.mark] = True

# Prepare array
tokens.sort()
tokens = ["SIL"] + tokens

print(tokens)