# pyright : reportOperatorIssue = false

import torch
import triton
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import cast

from torch.nn.attention import SDPBackend, _sdpa_kernel_variadic

from forward.kernel_fa2_forward import fa2_forward
import math

def benchmark_forward():
    """
    Benchmark the forward of the Flash Attention 2 algorithm.
    """
    H = 1
    B = 1
    d = 64
    sizes_N = [64, 128, 256, 512, 1024, 2048, 4096]
    
    dtype = torch.bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dict = {"size_N" : [], "kernel" : [], "pytorch" : []}
    
    for N in sizes_N:
        q_tensor = torch.randn((H, B, N, d), dtype=dtype, device=device)
        k_tensor = torch.randn((H, B, N, d), dtype=dtype, device=device)
        v_tensor = torch.randn((H, B, N, d), dtype=dtype, device=device)
        
        ms_kernel = cast(float, triton.testing.do_bench(
            lambda : fa2_forward(q_tensor, k_tensor, v_tensor),
            warmup=25,
            rep=100,
            return_mode='median'
        ))
        
        with _sdpa_kernel_variadic(SDPBackend.FLASH_ATTENTION):
            ms_torch = cast(float, triton.testing.do_bench(
                lambda : F.scaled_dot_product_attention(q_tensor, k_tensor, v_tensor),
                warmup=25,
                rep=100,
                return_mode='median'
            ))
            
        bytes_transferred = math.ceil((N/64)) * (4*64*d + 4*N*d)
        
        brandwidth_kernel = (bytes_transferred / (ms_kernel)) * 1e-6
        brandwidth_torch = (bytes_transferred / (ms_torch)) * 1e-6 
        
        dict["size_N"].append(N)
        dict["kernel"].append(brandwidth_kernel)
        dict["pytorch"].append(brandwidth_torch)
    
    
    PEAK_BANDWIDTH = 896  # GB/s
    
    plt.plot(dict["size_N"], dict["kernel"], marker="o",label="Custom Triton Kernel")
    plt.plot(dict["size_N"], dict["pytorch"], marker="s", label="Pytorch Implementation")
    plt.axhline(PEAK_BANDWIDTH, linestyle="--", color="grey", label = f"RTX 5070 Ti Peak bandwidth ({PEAK_BANDWIDTH} GB/s) ")
    
    plt.xscale("log", base=2)
    plt.yscale("log")
    
    plt.xlabel("Matrix dimension (N)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("Custom kernel vs pytorch Flash Attention 2 - RTX 5070 Ti, BF16")
    
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("../fa2-triton-study/benchmark/figures/forward_v1.png", dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    benchmark_forward()