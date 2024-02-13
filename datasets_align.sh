set -e

# Download models
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
mfa model download acoustic russian_mfa
mfa model download dictionary russian_mfa
mfa model download acoustic ukrainian_mfa
mfa model download dictionary ukrainian_mfa

# Process VCTK
mfa align "$PWD/datasets/vctk-prepared" english_mfa english_mfa "$PWD/datasets/vctk-aligned" -t "$PWD/.mfa/" -j 16 --clean

# Process LibriTTS
mfa align "$PWD/datasets/libritts-prepared" english_us_mfa english_mfa "$PWD/datasets/libritts-aligned" -t "$PWD/.mfa/" -j 16 --clean

# Process Common Voice EN
# mfa validate "$PWD/datasets/common-voice-en-prepared" english_us_mfa english_mfa -t "$PWD/.mfa/" -j 16
# mfa align "$PWD/datasets/common-voice-en-prepared" english_us_mfa english_mfa "$PWD/datasets/common-voice-en-aligned" -t "$PWD/.mfa/" -j 16

# Process Common Voice RU
# mfa validate "$PWD/datasets/common-voice-ru-prepared" russian_mfa russian_mfa -t "$PWD/.mfa/" -j 16
# mfa align "$PWD/datasets/common-voice-ru-prepared" russian_mfa russian_mfa "$PWD/datasets/common-voice-ru-aligned" -t "$PWD/.mfa/" -j 16 --clean

# Process Common Voice UK
# mfa validate "$PWD/datasets/common-voice-uk-prepared" ukrainian_mfa ukrainian_mfa -t "$PWD/.mfa/" -j 16
# mfa align "$PWD/datasets/common-voice-uk-prepared" ukrainian_mfa ukrainian_mfa "$PWD/datasets/common-voice-uk-aligned" -t "$PWD/.mfa/" -j 16 --clean