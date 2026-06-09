import torch
import triton.language as tl
import pytest

import torch.nn.functional as F
from forward.kernel_fa2_forward import fa2_forward
from backward.kernel_fa2_backward import fa2_backward

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

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_backward_fkash_attention2(dtype):
    N = 16
    d = 16
    
    q_tensor = torch.randn((N, d), dtype=dtype, device=device, requires_grad=True)
    k_tensor = torch.randn((N, d), dtype=dtype, device=device, requires_grad=True)
    v_tensor = torch.randn((N, d), dtype=dtype, device=device, requires_grad=True)
    
    o_tensor, L_tensor = fa2_forward(q_tensor, k_tensor, v_tensor)
    
    o_torch = F.scaled_dot_product_attention(q_tensor, k_tensor, v_tensor)
    
    torch.testing.assert_close(o_tensor, o_torch, atol=1e-2, rtol=1e-2)
    
    do_tensor = torch.randn((N, d), dtype=dtype, device=device, requires_grad=True)
    
    D_tensor, dq_tensor, dk_tensor, dv_tensor = fa2_backward(q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, L_tensor)
    
    grad_q, grad_k, grad_v = torch.autograd.grad(
        outputs=o_torch,
        inputs=[q_tensor, k_tensor, v_tensor],
        grad_outputs=do_tensor
    )
    
    torch.testing.assert_close(grad_q, dq_tensor)