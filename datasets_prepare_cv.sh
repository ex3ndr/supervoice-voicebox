set -e
mfa model download acoustic english_mfa
mfa model download dictionary english_us_mfa
mfa validate "$PWD/datasets/cv-en-prepared" english_us_mfa english_mfa -t "$PWD/.mfa/" -j 16
mfa align "$PWD/datasets/cv-en-prepared" english_us_mfa english_mfa "$PWD/datasets/cv-en-aligned" -t "$PWD/.mfa/" -j 16
