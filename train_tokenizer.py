from glob import glob
from tqdm import tqdm
import textgrid
import requests
import json

# Parameters

# Ignoring modifiers
ignore = [
    # 'ː', 'ˑ', '◌̆', # This is a length-based modifier and should be ignored since we have a separate duration model

    # "˥", "˧", "˩", '˦', '˨', # Tonal modifiers doesn't make sence for non-tonal languages (croatian, czech, etc.)
    
    # 'ʷ', 'ʲ' # These are palatalization and labialization modifiers, which are not used in the languages we are interested in
]

mfa_links = [
    'english/mfa/v2.2.1',
    'english/us_mfa/v2.2.1',
    'english/uk_mfa/v2.2.1',
    'english/nigeria_mfa/v2.2.1',
    'english/india_mfa/v2.2.1',
    'bulgarian/mfa/v2.0.0a',
    'croatian/mfa/v2.0.0a',
    'czech/mfa/v2.0.0a',
    'french/mfa/v2.0.0a',
    'german/mfa/v2.0.0a',
    'hausa/mfa/v2.0.0a',
    'japanese/mfa/v2.0.1a',
    'korean/jamo_mfa/v2.0.0',
    'mandarin/china_mfa/v2.0.0a',
    'mandarin/erhua_mfa/v2.0.0a',
    'mandarin/mfa/v2.0.0a',
    'mandarin/taiwan_mfa/v2.0.0a',
    'polish/mfa/v2.0.0a',
    'portuguese/brazil_mfa/v2.0.0a',
    'portuguese/mfa/v2.0.0a',
    'portuguese/portugal_mfa/v2.0.0a',
    'russian/mfa/v2.0.0a',
    'spanish/latin_america_mfa/v2.0.0a',
    'spanish/spain_mfa/v2.0.0a',
    'spanish/mfa/v2.0.0a',
    'swahili/mfa/v2.0.0a',
    'swedish/mfa/v2.0.0a',
    'tamil/mfa/v2.0.0',
    'thai/mfa/v2.0.0a',
    'turkish/mfa/v2.0.0a',
    'ukrainian/mfa/v2.0.0a',
    'vietnamese/hanoi_mfa/v2.0.0a',
    'vietnamese/ho_chi_minh_city_mfa/v2.0.0a',
    'vietnamese/hue_mfa/v2.0.0',
    'vietnamese/mfa/v2.0.0a'
]

# Loaders
def download_mfa_phones(id):
    meta =  requests.get("https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/dictionary/" + id + "/meta.json", allow_redirects=True).json()

    return meta['phones'], meta['name']

# Load phones
existing = {}
tokens = []
for link in mfa_links:
    ph, n = download_mfa_phones(link)

    for p in ph:

        # Remove ignored
        for i in ignore:
            p = p.replace(i, '')

        # Add
        if p not in existing:
            existing[p] = [n]
            tokens.append(p)
        else:
            existing[p].append(n)

# Prepare array
tokens.sort()

# Print results
print(tokens)
# print(ignore)
# for t in tokens:
#     print(t, existing[t])
