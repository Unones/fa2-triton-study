import torch
import triton.language as tl
import pytest

import torch.nn.functional as F
from forward.kernel_fa2_forward import fa2_forward
from forward.ref2_fa2_forward import ref2_fa2_forward

torch_to_triton_dtypes = {
    torch.float32 : tl.float32,
    torch.float16 : tl.float16,
    torch.bfloat16 : tl.bfloat16
}

tols_dtypes = {
    torch.float32 : {"atol" : 1e-5, "rtol" : 1e-5},
    torch.float16 : {"atol" : 1e-3, "rtol" : 1e-3},
    torch.bfloat16 : {"atol" : 1e-2, "rtol" : 1e-2}
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@pytest.mark.parametrize("dtype" , [torch.float32, torch.float16, torch.bfloat16])
def test_forward_flash_attention2(dtype):
    N = 10
    d = 50

    q_tensor = torch.randn((N, d), dtype=dtype, device=device)
    k_tensor = torch.randn((N, d), dtype=dtype, device=device)
    v_tensor = torch.randn((N, d), dtype=dtype, device=device)
    
    o_tensor, _ = fa2_forward(q_tensor, k_tensor, v_tensor, torch_to_triton_dtypes[dtype])
    
    o_torch = F.scaled_dot_product_attention(q_tensor, k_tensor, v_tensor)
    
    o_ref, _ = ref2_fa2_forward(q_tensor, k_tensor, v_tensor)
    
    torch.testing.assert_close(o_tensor, o_ref, atol = 1e-3, rtol=1e-3)