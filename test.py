    dims = results["dim"]
    tflops_kernel_v1 = results["kernel_v1"]
    tflops_kernel_v2 = results["kernel_v2"]
    tflops_kernel_v3 = results["kernel_v3"]
    tflops_matmul = results["torch"]

    PEAK_TFLOPS = 43.9  # RTX 5070 Ti FP32

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(dims, tflops_kernel_v1, marker="o", label="Custom Triton kernel (v1_baseline)")
    ax.plot(dims, tflops_kernel_v2, marker="o", label="Custom Triton kernel (v2_group_ordering)")
    ax.plot(dims, tflops_kernel_v3, marker="o", label="Custom Triton kernel (v3_autotuning)")
    ax.plot(dims, tflops_matmul, marker="s", label="torch.matmul (cuBLAS)")
    ax.axhline(PEAK_TFLOPS, linestyle="--", color="grey", label=f"Peak FP32 ({PEAK_TFLOPS} TFLOPS)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    ax.set_xlabel("Matrix dimension (M = N = P)")
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.set_title("Custom kernels matmul vs cuBLAS — RTX 5070 Ti, FP32")

    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)