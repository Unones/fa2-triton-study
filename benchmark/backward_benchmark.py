# pyright : reportOperatorIssue = false

import torch
import triton
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import cast

from torch.nn.attention import SDPBackend, _sdpa_kernel_variadic

from forward.kernel_fa2_forward import fa2_forward
from backward.kernel_fa2_backward import fa2_backward

def benchmark_forward():
    """
    Benchmark the forward of the Flash Attention 2 algorithm.
    """
    H = 32
    B = 32
    d = 64
    sizes_N = [64, 128, 256, 512, 1024, 2048, 4096]
    
    dtype = torch.bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dict = {"size_N" : [], "kernel" : [], "pytorch" : []}
    
    for N in sizes_N:
        q_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device, requires_grad=True)
        k_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device, requires_grad=True)
        v_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device, requires_grad=True)
        do_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device, requires_grad=True)
        
        o_tensor, L_tensor = fa2_forward(q_tensor, k_tensor, v_tensor)
        
        ms_kernel = cast(float, triton.testing.do_bench(
            lambda : fa2_backward(q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, L_tensor),
            warmup=25,
            rep=100,
            return_mode='median'
        ))
        
        o_torch = F.scaled_dot_product_attention(q_tensor, k_tensor, v_tensor)
        
        with _sdpa_kernel_variadic(SDPBackend.FLASH_ATTENTION):
            ms_torch = cast(float, triton.testing.do_bench(
                lambda : torch.autograd.grad(
                    outputs=o_torch,
                    inputs=[q_tensor, k_tensor, v_tensor],
                    grad_outputs=do_tensor,
                    retain_graph=True,
                ),
                warmup=25,
                rep=100,
                return_mode='median'
            ))
            
        flops = 10*B*H*N*N*d
        
        flop_kernel = (flops / (ms_kernel )) * 1e-9
        flop_torch = (flops / (ms_torch )) * 1e-9
        
        dict["size_N"].append(N)
        dict["kernel"].append(flop_kernel)
        dict["pytorch"].append(flop_torch)
    
    PEAK_TFLOPS = 87.9 *(2.30 / 2.452)     #TFLOP/s
    
    plt.plot(dict["size_N"], dict["kernel"], marker="o",label="Custom Triton Kernel")
    plt.plot(dict["size_N"], dict["pytorch"], marker="s", label="Pytorch Implementation")
    plt.axhline(PEAK_TFLOPS, linestyle="--", color="grey", label = f"RTX 5070 Ti Peak throughput ({PEAK_TFLOPS:.1f} TFLOP/s) ")
    
    plt.xscale("log", base=2)
    plt.yscale("log")
    
    plt.xlabel("Matrix dimension (N)")
    plt.ylabel("Throughput (TFLOP/s)")
    plt.title("Backward Pass Flash Attention 2 - RTX 5070 Ti, BF16")
    
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("../fa2-triton-study/benchmark/backward_4dims.png", dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    benchmark_forward()