set -e

# Download models
mfa model download acoustic english_mfa
mfa model download dictionary english_us_mfa

# Process VCTK
mfa validate "$PWD/datasets/vctk-prepared" english_us_mfa english_mfa -t "$PWD/.mfa/" -j 16
mfa align "$PWD/datasets/vctk-prepared" english_us_mfa english_mfa "$PWD/datasets/vctk-aligned" -t "$PWD/.mfa/" -j 16

# Process LibriTTS
mfa validate "$PWD/datasets/libritts-prepared" english_us_mfa english_mfa -t "$PWD/.mfa/" -j 16
mfa align "$PWD/datasets/libritts-prepared" english_us_mfa english_mfa "$PWD/datasets/libritts-aligned" -t "$PWD/.mfa/" -j 16

# Process Common Voice EN
mfa validate "$PWD/datasets/common-voice-en-prepared" english_us_mfa english_mfa -t "$PWD/.mfa/" -j 16
mfa align "$PWD/datasets/common-voice-en-prepared" english_us_mfa english_mfa "$PWD/datasets/common-voice-en-aligned" -t "$PWD/.mfa/" -j 16