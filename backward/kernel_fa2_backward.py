import triton
import triton.language as tl
import torch
import math

import torch.nn.functional as F
from forward.kernel_fa2_forward import fa2_forward

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
    L_ptr, D_ptr,
    stride_q_row,
    stride_k_col,
    stride_v_col,
    stride_o_row,
    stride_do_row,
    size_n, d, sqrt_d, 
    size_d : tl.constexpr,
    output_dtype : tl.constexpr,
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
    
    offset_k = offset_k_col[:, None] + offset_d[None, :]
    offset_v = offset_v_col[:, None] + offset_d[None, :]
    
    mask_kv = mask_col[:, None] & mask_d[None, :]
    
    k = tl.load(k_ptr + offset_k, mask=mask_kv, other=0).to(dtype=tl.float32)
    v = tl.load(v_ptr + offset_v, mask=mask_kv, other=0).to(dtype=tl.float32)
    
    dk = tl.zeros((BS_col, size_d), dtype=tl.float32)
    dv = tl.zeros((BS_col, size_d), dtype=tl.float32)
    
    for i in range(1, nb_blocks_row +1):
        offset_row = i*BS_row + tl.arange(0, BS_row)
        
        offset_q_row = offset_row * stride_q_row
        offset_do_row = offset_row * stride_do_row
        
        mask_row = offset_row < size_n
        
        offset_q = offset_q_row[:, None] + offset_d[None, :]
        offset_do = offset_do_row[:, None] + offset_d[None, :]
        offset_dq = offset_q_row[:, None] + offset_d[None, :]
        
        mask_qo = mask_row[:, None] & mask_d[None, :]
        
        q = tl.load(q_ptr + offset_q, mask=mask_qo, other=0).to(dtype=tl.float32)
        do = tl.load(do_ptr + offset_do, mask=mask_qo, other=0).to(dtype=tl.float32)
        L_row = tl.load(L_ptr + offset_row, mask=mask_row, other=0)
        D_row = tl.load(D_ptr + offset_row, mask=mask_row, other=0)
        
        k_t = tl.trans(k)
        s = tl.dot(q, k_t) / sqrt_d
        
        mask_s = mask_row[:, None] & mask_col[None, :]
        
        p_intermediate_matrix = tl.where(mask_s, s - L_row[:, None], -float("inf"))
        p = tl.exp(p_intermediate_matrix)
        
        p_t = tl.trans(p)
        v_t = tl.trans(v)
        
        dv += tl.dot(p_t, do)
        dp = tl.dot(do, v_t)
        
        ds = p * (dp - D_row[:, None])
        dq = tl.dot(ds, k)
        dq = dq.to(dtype=output_dtype)
        tl.atomic_add(dq_ptr + offset_dq, dq)
        
        dk += tl.dot(ds, q)
    
    dk = dk.to(dtype=output_dtype)
    dv = dv.to(dtype=output_dtype)
    
    tl.store(dk_ptr + offset_k, dk, mask=mask_kv)
    tl.store(dv_ptr + offset_v, dv, mask=mask_kv)
        
        
torch_to_triton_dtypes = {
    torch.float32 : tl.float32,
    torch.float16 : tl.float16,
    torch.bfloat16 : tl.bfloat16
}    

    
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
    stride_q_row = q_tensor.stride(0)
    stride_k_col = k_tensor.stride(0)
    stride_v_col = v_tensor.stride(0)
    
    D_tensor = torch.empty(size=(N,), dtype=dtype, device=device)
    dq_tensor = torch.empty(size=(N, d), dtype=dtype, device=device)
    dk_tensor = torch.empty(size=(N, d), dtype=dtype, device=device)
    dv_tensor = torch.empty(size=(N, d), dtype=dtype, device=device)
    
    BS_row = 16
    BS_col = 16
    
    nb_tiles_row = math.ceil(N / BS_row)
    nb_tiles_col = math.ceil(N / BS_col)
    
    grid_D = (nb_tiles_row,)
    
    triton_dtype = torch_to_triton_dtypes[dtype]
    
    args_D = (
        o_tensor, do_tensor,
        D_tensor,
        N, d, size_d, 
        stride_o_row,
        stride_do_row,
        triton_dtype,
        BS_row
    )
    
    _kernel_D_fa2[grid_D](*args_D)      #type:ignore
    
    sqrt_d = math.sqrt(d)
    size_d = triton.next_power_of_2(d)
    
    grid_backward = (nb_tiles_col, )
    
    args_backward = (
        q_tensor, dq_tensor,
        k_tensor, dk_tensor,
        v_tensor, dv_tensor,
        o_tensor, do_tensor,
        L_tensor, D_tensor,
        stride_q_row,
        stride_k_col,
        stride_v_col,
        stride_o_row,
        stride_do_row,
        N, d, sqrt_d, size_d,
        triton_dtype, nb_tiles_row,
        BS_row, BS_col
    )
    
    _kernel_fa2_backward[grid_backward](*args_backward) #type:ignore
    
    
    return D_tensor, dq_tensor, dk_tensor, dv_tensor
    

if __name__ == "__main__":
    N = 2
    d = 10
    
    dtype=torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
    
    
    
    print(f"The tensor dq_tensor calculated by the kernel is equal to : \n {dq_tensor}")
    print(f"The tensor grad_q calculated by pytorch is equal to : \n {grad_q}")
    