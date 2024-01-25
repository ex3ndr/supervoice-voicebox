docker run \
    --gpus '"all"' \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --rm  \
    -v .:/data/ \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    -e "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256" \
    -e "PYTHONIOENCODING=utf8" \
    -it winglian/axolotl:main-py3.10-cu118-2.0.1 \
    accelerate launch -m axolotl.cli.train /data/train_gpt.yaml