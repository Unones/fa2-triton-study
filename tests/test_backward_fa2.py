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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _warmup_autograd():
    a = torch.randn(8, 8, device="cuda", requires_grad=True)
    b = torch.randn(8, 8, device="cuda", requires_grad=True)
    (a @ b).sum().backward()   # force l'init du contexte sur le thread autograd
    torch.cuda.synchronize()
    

@pytest.mark.parametrize("dtype" , [torch.bfloat16])
@pytest.mark.parametrize("H", [8, 10, 16, 45, ])
@pytest.mark.parametrize("B", [8, 10, 16, 45,])
@pytest.mark.parametrize("N", [8, 10, 100, 128])
@pytest.mark.parametrize("d", [16, 20, 64])
def test_backward_fkash_attention2(dtype,B, H, N, d):
    
    _warmup_autograd()

    q_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device, requires_grad=True)
    k_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device, requires_grad=True)
    v_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device, requires_grad=True)
    
    o_tensor, L_tensor = fa2_forward(q_tensor, k_tensor, v_tensor)
    
    o_torch = F.scaled_dot_product_attention(q_tensor, k_tensor, v_tensor)
    
    torch.testing.assert_close(o_tensor, o_torch, atol=1e-2, rtol=1e-2)
    
    do_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device, requires_grad=True)
    
    _, dq_tensor, dk_tensor, dv_tensor = fa2_backward(q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, L_tensor)
    
    grad_q, grad_k, grad_v = torch.autograd.grad(
        outputs=o_torch,
        inputs=[q_tensor, k_tensor, v_tensor],
        grad_outputs=do_tensor
    )
    
    torch.testing.assert_close(grad_q, dq_tensor, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(grad_k, dk_tensor, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(grad_v, dv_tensor, atol=1e-2, rtol=1e-2)
    