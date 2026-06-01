import triton
import triton.language as tl
import torch
import math

from backward.ref_fa2_D_tensor import ref_D_tensor

@triton.jit
def _kernel_D_fa2(
    o_ptr, do_ptr,
    D_ptr,
    size_n, d, size_d : tl.constexpr,
    stride_o_row,
    stride_do_row,
    output_dtype : tl.constexpr,
    BS_row : tl.constexpr
):
    """
    Computes the D vector for the backward of flash-attention 2.
    
    """
    
    row_index = tl.program_id(0)
    
    offset_row = row_index * BS_row + tl.arange(0, BS_row)
    offset_o_row = offset_row * stride_o_row
    offset_do_row = offset_row * stride_do_row
    
    mask_row = offset_row < size_n
    
    offset_d = tl.arange(0, size_d)
    mask_d = offset_d < d
    
    offset_o = offset_o_row[:, None] + offset_d[None, :]
    offset_do = offset_do_row[:, None] + offset_d[None, :]
    
    mask = mask_row[:, None] & mask_d[None, :]
    
    o = tl.load(o_ptr + offset_o, mask=mask, other=0).to(tl.float32)
    do = tl.load(do_ptr + offset_do, mask=mask, other=0).to(tl.float32)
    
    pointwise_mul = o * do
    D_mat = tl.sum(pointwise_mul, axis=1)
    D_mat = D_mat.to(dtype=output_dtype)
    
    tl.store(D_ptr + offset_row, D_mat, mask=mask_row)
    


@triton.jit
def _kernel_fa2_backward(
    q_ptr, dq_ptr,
    k_ptr, dk_ptr,
    v_ptr, dv_ptr, 
    o_ptr, do_ptr,
    L_ptr,
    stride_q_row,
    stride_k_col,
    stride_v_col,
    stride_o_row,
    stride_do_row,
    size_n, d, sqrt_d, size_d,
    nb_blocks_row,
    BS_row : tl.constexpr,
    BS_col : tl.constexpr
):
    """
    Computes the backward gradients of the attention mechanism.
    
    It uses the Flash Attention 2 implementation.
    
    
    """
    
    col_index = tl.program_id(0)
    
    offset_col = col_index * BS_col + tl.arange(0, BS_col)
    offset_k_col = offset_col * stride_k_col
    offset_v_col = offset_col * stride_v_col
    
    mask_col = offset_col < size_n
    
    offset_d = tl.arange(0, size_d)
    mask_d = offset_d < d
    
    
    
    
def fa2_backward(
    q_tensor : torch.Tensor,
    k_tensor : torch.Tensor,
    v_tensor : torch.Tensor,
    o_tensor : torch.Tensor,
    do_tensor : torch.Tensor,
    L_tensor : torch.Tensor
):
    """
    Computes the backward gradients of the attention mechanism.
    
    """
    
    N, d = q_tensor.shape
    dtype = q_tensor.dtype
    device = q_tensor.device
    
    size_d = triton.next_power_of_2(d)
    
    q_tensor = q_tensor.contiguous()
    k_tensor = k_tensor.contiguous()
    v_tensor = v_tensor.contiguous()
    o_tensor = o_tensor.contiguous()
    do_tensor = do_tensor.contiguous()
    
    stride_o_row = o_tensor.stride(0)
    stride_do_row = o_tensor.stride(0)
    
    D_tensor = torch.empty(size=(N,), dtype=dtype, device=device)
    
    BS_row = 4
    BS_col = 4
    
    nb_tiles_row = math.ceil(N / BS_row)
    nb_tiles_col = math.ceil(N / BS_col)
    
    grid_D = (nb_tiles_row,)
    
    args_D = (
        o_tensor, do_tensor,
        D_tensor,
        N, d, size_d, 
        stride_o_row,
        stride_do_row,
        tl.float32,
        BS_row
    )
    
    _kernel_D_fa2[grid_D](*args_D)      #type:ignore
    
    return D_tensor
    

if __name__ == "__main__":
    N = 6
    d = 100
    
    dtype = torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    q_tensor = torch.randn((N, d), dtype=dtype, device=device)
    k_tensor = torch.randn((N, d), dtype=dtype, device=device)
    v_tensor = torch.randn((N, d), dtype=dtype, device=device)
    o_tensor = torch.randn((N, d), dtype=dtype, device=device)
    do_tensor = torch.randn((N, d), dtype=dtype, device=device)

    L_tensor = torch.randn((N,), dtype=dtype, device=device)
    
    D_tensor = fa2_backward(q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, L_tensor)
    D_ref = ref_D_tensor(o_tensor, do_tensor, dtype)
    
    torch.testing.assert_close(D_tensor, D_ref, atol=1e-5, rtol=1e-5)
    
    