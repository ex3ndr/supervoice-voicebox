set -e
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
accelerate launch ./train.py --yaml "$1"
# while true; do
#     accelerate launch ./train.py --yaml "$1" || true
# done