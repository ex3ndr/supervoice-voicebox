docker run \
    --gpus '"all"' \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --rm  \
    -v .:/data/ \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    -e "CUDA_VISIBLE_DEVICES=0" \
    -e "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256" \
    -it winglian/axolotl:main-py3.10-cu118-2.0.1 \
    accelerate launch -m axolotl.cli.inference /data/train_gpt.yaml --lora_model_dir="/data/gpt4/checkpoint-276"