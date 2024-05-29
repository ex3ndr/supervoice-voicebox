set -e
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024'
# accelerate launch ./train.py
while true; do
    accelerate launch ./train.py || true
done