import math
import torch

#
# This is a symmetric version of the Alibi
# Source: https://github.com/ofirpress/attention_with_linear_biases/issues/5
#

slope_cache = {}
def get_slopes(n, device = "cpu"):

    # Check cache
    global slope_cache
    key = f"{str(n)}-{str(device)}"
    if key in slope_cache:
        return slope_cache[key]

    # Only powers of two are supported: provided code has a bug that produces weird results for other values and i don't want to fix it
    # since i am not going to use it anyway
    if not math.log2(n).is_integer():
        raise ValueError("n must be a power of 2")

    # Calculate slope
    start = (2**(-2**-(math.log2(n)-3)))
    ratio = start
    tensor = torch.Tensor([start*ratio**i for i in range(n)]) * -1
    tensor = tensor.to(device)

    # Cache and return
    slope_cache[key] = tensor
    return tensor