import torch
from flash_attn import flash_attn_func

#
# Implementations
#

def do_attention(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)

def do_attention_pytorch(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)

def do_attention_manual(q, k, v):
    y = torch.matmul(q, k.transpose(-2, -1)) # (B, H, T, T)
    y = y / torch.sqrt(torch.tensor(q.shape[-1], dtype = y.dtype, device = y.device)) # (B, H, T, T)
    y = torch.nn.functional.softmax(y, dim = -1) # (B, H, T, T)
    y = torch.matmul(y, v) # (B, H, T, C)
    return y

def do_attention_flash(q, k, v):
    return flash_attn_func(q, k, v)

#
# Unit Tests
#

def test_do_attention():
    # Arrange
    q = torch.rand(1, 10, 16, 10).to(torch.bfloat16).cuda()
    k = torch.rand(1, 10, 16, 10).to(torch.bfloat16).cuda()
    v = torch.rand(1, 10, 16, 10).to(torch.bfloat16).cuda()

    # Act
    y1 = do_attention_manual(q, k, v)
    y2 = do_attention_pytorch(q, k, v)
    y3 = do_attention_flash(q, k, v)

    print(y1, y2, y3)

    # Assert
    assert y1.shape == (1, 10, 16, 10)
    assert y2.shape == (1, 10, 16, 10)
    assert torch.allclose(y1, y2)
    assert torch.allclose(y1, y3)
    assert torch.allclose(y2, y3)

#
# Run Tests
#

if __name__ == '__main__':
    test_do_attention()