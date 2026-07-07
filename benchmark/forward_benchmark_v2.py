# pyright : reportOperatorIssue = false

import torch
import triton
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.nn.attention import SDPBackend, _sdpa_kernel_variadic

from forward.kernel_fa2_forward import fa2_forward

def benchmark_forward():
    """
    Benchmark the forward of the Flash Attention 2 algorithm.
    """
    H = 32
    B = 32
    d = 64
    sizes_N = [64, 128, 256, 512, 1024, 2048, 4096]
    
    dtype = torch.float16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dict = {"size_N" : [], "kernel" : [], "pytorch" : []}
    
    for N in sizes_N:
        q_tensor = torch.randn((H, B, N, d), dtype=dtype, device=device)
        k_tensor = torch.randn((H, B, N, d), dtype=dtype, device=device)
        v_tensor = torch.randn((H, B, N, d), dtype=dtype, device=device)
        
        ms_kernel = triton.testing.do_bench(
            lambda : fa2_forward(q_tensor, k_tensor, v_tensor),
            warmup=25,
            rep=100,
            return_mode='median'
        )
        
        with _sdpa_kernel_variadic(SDPBackend.FLASH_ATTENTION):
            ms_torch = triton.testing.do_bench(
                lambda : F.scaled_dot_product_attention(q_tensor, k_tensor, v_tensor),
                warmup=25,
                rep=100,
                return_mode='median'
            )
            
        flops = 4*N*N*d*H*B
        
        flop_kernel = (flops / (ms_kernel )) * 1e-9
        flop_torch = (flops / (ms_torch )) * 1e-9
        
        dict["size_N"].append(N)
        dict["kernel"].append(flop_kernel)
        dict["pytorch"].append(flop_torch)
    
    PEAK_TFLOPS = 87.9      #TFLOP/s
    
    plt.plot(dict["size_N"], dict["kernel"], marker="o",label="Custom Triton Kernel")
    plt.plot(dict["size_N"], dict["pytorch"], marker="s", label="Pytorch Implementation")
    plt.axhline(PEAK_TFLOPS, linestyle="--", color="grey", label = f"RTX 5070 Ti Peak bandwidth ({PEAK_TFLOPS} TFLOP/s) ")
    
    plt.xscale("log", base=2)
    plt.yscale("log")
    
    plt.xlabel("Matrix dimension (N)")
    plt.ylabel("Throughput (TFLOP/s)")
    plt.title("Custom kernel vs pytorch Flash Attention 2 - RTX 5070 Ti, FP16")
    
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("../fa2-triton-study/benchmark/forward_4dims.png")


if __name__ == "__main__":
    benchmark_forward()