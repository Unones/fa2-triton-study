import torch
from forward.kernel_fa2_forward import fa2_forward

import torch.nn.functional as F
from torch.nn.attention import SDPBackend, _sdpa_kernel_variadic


def main():
    N = 4096
    d = 64

    dtype = torch.bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    q_tensor = torch.randn((N, d), dtype=dtype, device=device)
    k_tensor = torch.randn((N, d), dtype=dtype, device=device)
    v_tensor = torch.randn((N, d), dtype=dtype, device=device)

    q_tensor_torch = q_tensor.reshape(1, 1, N, d)
    k_tensor_torch = k_tensor.reshape(1, 1, N, d)
    v_tensor_torch = v_tensor.reshape(1, 1, N, d)

    # Warmup (hors zone profilée)
    fa2_forward(q_tensor, k_tensor, v_tensor)
    with _sdpa_kernel_variadic(SDPBackend.FLASH_ATTENTION):
        F.scaled_dot_product_attention(q_tensor_torch, k_tensor_torch, v_tensor_torch)
    torch.cuda.synchronize()

    # Zone profilée
    torch.cuda.cudart().cudaProfilerStart() #type:ignore
    fa2_forward(q_tensor, k_tensor, v_tensor)
    with _sdpa_kernel_variadic(SDPBackend.FLASH_ATTENTION):
        F.scaled_dot_product_attention(q_tensor_torch, k_tensor_torch, v_tensor_torch)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()  #type:ignore


if __name__ == "__main__":
    main()