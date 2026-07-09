import torch
from forward.kernel_fa2_forward import fa2_forward
from backward.kernel_fa2_backward import fa2_backward

import torch.nn.functional as F
from torch.nn.attention import SDPBackend, _sdpa_kernel_variadic


def main():
    H = 32
    B = 32
    N = 4096
    d = 64

    dtype = torch.bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    q_tensor = torch.randn((H, B, N, d), dtype=dtype, device=device, requires_grad=True)
    k_tensor = torch.randn((H, B, N, d), dtype=dtype, device=device, requires_grad=True)
    v_tensor = torch.randn((H, B, N, d), dtype=dtype, device=device, requires_grad=True)
    do_tensor = torch.randn((H, B, N, d), dtype=dtype, device=device, requires_grad=True)
    
    o_tensor, L_tensor = fa2_forward(q_tensor, k_tensor, v_tensor)
    o_torch = F.scaled_dot_product_attention(q_tensor, k_tensor, v_tensor)

    # Warmup (hors zone profilée)
    fa2_backward(q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, L_tensor)
    with _sdpa_kernel_variadic(SDPBackend.FLASH_ATTENTION):
        torch.autograd.grad(
            outputs=o_torch,
            inputs=[q_tensor, k_tensor, v_tensor],
            grad_outputs=do_tensor,
            retain_graph=True,
        )
    torch.cuda.synchronize()

    # Zone profilée
    torch.cuda.cudart().cudaProfilerStart() #type:ignore
    fa2_backward(q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, L_tensor)
    with _sdpa_kernel_variadic(SDPBackend.FLASH_ATTENTION):
        torch.autograd.grad(
            outputs=o_torch,
            inputs=[q_tensor, k_tensor, v_tensor],
            grad_outputs=do_tensor,
            retain_graph=True,
        )
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()  #type:ignore


if __name__ == "__main__":
    main()