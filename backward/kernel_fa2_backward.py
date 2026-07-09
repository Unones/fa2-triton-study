# import os
# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import triton
import triton.language as tl
import torch
import math

import torch.nn.functional as F
from forward.kernel_fa2_forward import fa2_forward
from backward.ref_fa2_D_tensor import ref_D_tensor

@triton.autotune(configs=[
    triton.Config(kwargs={"BS_row" : 32}, num_stages=2),
    triton.Config(kwargs={"BS_row" : 32}, num_stages=3),
    triton.Config(kwargs={"BS_row" : 32}, num_stages=4),
    triton.Config(kwargs={"BS_row" : 64}, num_stages=2),
    triton.Config(kwargs={"BS_row" : 64}, num_stages=3),
    triton.Config(kwargs={"BS_row" : 64}, num_stages=4),
],
    key=["size_n"],)
@triton.jit
def _kernel_D_fa2(
    o_ptr, do_ptr,
    D_ptr,
    size_n, d, size_d : tl.constexpr,
    stride_o_batch,
    stride_do_batch,
    stride_D_batch,
    stride_o_row,
    stride_do_row,
    BS_row : tl.constexpr
):
    """
    Computes the D vector for the backward of flash-attention 2.
    
    """
    
    BS_batch = 1
    
    pid_row = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    offset_row = pid_row * BS_row + tl.arange(0, BS_row)
    offset_batch = pid_batch * BS_batch
    
    offset_o_row = offset_row * stride_o_row
    offset_do_row = offset_row * stride_do_row
    
    offset_o_batch = offset_batch * stride_o_batch
    offset_do_batch = offset_batch * stride_do_batch
    offset_D_batch = offset_batch * stride_D_batch
    
    mask_row = offset_row < size_n
    
    offset_d = tl.arange(0, size_d)
    mask_d = offset_d < d
    
    offset_o = offset_o_row[:, None] + offset_d[None, :] + offset_o_batch
    offset_do = offset_do_row[:, None] + offset_d[None, :] + offset_do_batch
    offset_D = offset_row + offset_D_batch
    
    mask = mask_row[:, None] & mask_d[None, :]

    o = tl.load(o_ptr + offset_o, mask=mask, other=0).to(tl.float32)
    do = tl.load(do_ptr + offset_do, mask=mask, other=0).to(tl.float32)
    
    pointwise_mul = o * do
    D_mat = tl.sum(pointwise_mul, axis=-1)

    tl.store(D_ptr + offset_D, D_mat, mask=mask_row)
    


@triton.autotune(configs=[
    triton.Config(kwargs={"BS_row" : 16, "BS_col" : 16}, num_stages=3),
    triton.Config(kwargs={"BS_row" : 16, "BS_col" : 16}, num_stages=4),
    triton.Config(kwargs={"BS_row" : 16, "BS_col" : 16}, num_stages=5),
    triton.Config(kwargs={"BS_row" : 32, "BS_col" : 32}, num_stages=2),
    triton.Config(kwargs={"BS_row" : 32, "BS_col" : 32}, num_stages=3),
    triton.Config(kwargs={"BS_row" : 32, "BS_col" : 32}, num_stages=4),
    triton.Config(kwargs={"BS_row" : 32, "BS_col" : 64}, num_stages=3),
    triton.Config(kwargs={"BS_row" : 32, "BS_col" : 64}, num_stages=4),
    triton.Config(kwargs={"BS_row" : 64, "BS_col" : 64}, num_stages=2),
    triton.Config(kwargs={"BS_row" : 64, "BS_col" : 64}, num_stages=3),
    triton.Config(kwargs={"BS_row" : 64, "BS_col" : 64}, num_stages=4),
],
    key=["size_n"],
    reset_to_zero=["dq_ptr"])
@triton.jit
def _kernel_fa2_backward(
    q_ptr, dq_ptr,
    k_ptr, dk_ptr,
    v_ptr, dv_ptr, 
    do_ptr,
    L_ptr, D_ptr,
    stride_q_batch,
    stride_k_batch,
    stride_v_batch,
    stride_do_batch,
    stride_L_batch,
    stride_D_batch,
    stride_q_row,
    stride_k_col,
    stride_v_col,
    stride_do_row,
    size_n, d, sqrt_d, 
    size_d : tl.constexpr,
    output_dtype : tl.constexpr,
    BS_row : tl.constexpr,
    BS_col : tl.constexpr
):
    """
    Computes the backward gradients of the attention mechanism.
    
    It uses the Flash Attention 2 implementation.
    
    
    """
    
    nb_tiles_row = tl.cdiv(size_n, BS_row)
    
    BS_batch = 1
    
    pid_col = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    offset_col = pid_col * BS_col + tl.arange(0, BS_col)
    offset_batch = pid_batch * BS_batch
    
    offset_k_col = offset_col * stride_k_col
    offset_v_col = offset_col * stride_v_col
    
    offset_k_batch = offset_batch * stride_k_batch
    offset_v_batch = offset_batch * stride_v_batch
    offset_q_batch = offset_batch * stride_q_batch
    offset_do_batch = offset_batch * stride_do_batch
    offset_L_batch = offset_batch * stride_L_batch
    offset_D_batch = offset_batch * stride_D_batch

    mask_col = offset_col < size_n

    offset_d = tl.arange(0, size_d)
    mask_d = offset_d < d

    offset_k = offset_k_col[:, None] + offset_d[None, :] + offset_k_batch
    offset_v = offset_v_col[:, None] + offset_d[None, :] + offset_v_batch
    
    mask_kv = mask_col[:, None] & mask_d[None, :]
    
    k = tl.load(k_ptr + offset_k, mask=mask_kv, other=0)
    v = tl.load(v_ptr + offset_v, mask=mask_kv, other=0)
    
    dk = tl.zeros((BS_col, size_d), dtype=tl.float32)
    dv = tl.zeros((BS_col, size_d), dtype=tl.float32)

    for i in range(nb_tiles_row):
        offset_row = i*BS_row + tl.arange(0, BS_row)
        
        offset_q_row = offset_row * stride_q_row
        offset_do_row = offset_row * stride_do_row

        mask_row = offset_row < size_n
        
        offset_q = offset_q_row[:, None] + offset_d[None, :] + offset_q_batch
        offset_do = offset_do_row[:, None] + offset_d[None, :] + offset_do_batch
        offset_dq = offset_q_row[:, None] + offset_d[None, :] + offset_q_batch
        
        offset_L = offset_row + offset_L_batch
        offset_D = offset_row + offset_D_batch
        
        mask_qo = mask_row[:, None] & mask_d[None, :]
        
        q = tl.load(q_ptr + offset_q, mask=mask_qo, other=0)
        do = tl.load(do_ptr + offset_do, mask=mask_qo, other=0)
        L_row = tl.load(L_ptr + offset_L, mask=mask_row, other=0)
        D_row = tl.load(D_ptr + offset_D, mask=mask_row, other=0)

        k_t = tl.trans(k)
        s = tl.dot(q, k_t) / sqrt_d
        
        mask_s = mask_row[:, None] & mask_col[None, :]
        
        p_intermediate_matrix = tl.where(mask_s, s - L_row[:, None], -float("inf"))
        p = tl.exp(p_intermediate_matrix)
        
        p_t = tl.trans(p)
        v_t = tl.trans(v)
        
        dv += tl.dot(p_t.to(dtype=output_dtype), do)
        dp = tl.dot(do, v_t)
        
        ds = p * (dp - D_row[:, None])
        dq = tl.dot(ds.to(dtype=output_dtype), k) / sqrt_d
        
        tl.atomic_add(dq_ptr + offset_dq, dq)
        
        ds_t = tl.trans(ds)
        dk += tl.dot(ds_t.to(dtype=output_dtype), q) / sqrt_d
    
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
    
    batch, heads, N, d = q_tensor.shape
    
    dtype = q_tensor.dtype
    device = q_tensor.device
 
    q_tensor = q_tensor.contiguous()
    k_tensor = k_tensor.contiguous()
    v_tensor = v_tensor.contiguous()
    o_tensor = o_tensor.contiguous()
    do_tensor = do_tensor.contiguous()
    
    q_tensor = q_tensor.view(heads*batch, N, d)
    k_tensor = k_tensor.view(heads*batch, N, d)
    v_tensor = v_tensor.view(heads*batch, N, d)
    o_tensor = o_tensor.view(heads*batch, N, d)
    do_tensor = do_tensor.view(heads*batch, N, d)
    L_tensor = L_tensor.view(heads*batch, N)
    
    D_tensor = torch.empty(size=(heads*batch, N,), dtype=torch.float32, device=device)
    dq_tensor = torch.zeros(size=(heads*batch, N, d), dtype=torch.float32, device=device)
    dk_tensor = torch.empty(size=(heads*batch, N, d), dtype=torch.float32, device=device)
    dv_tensor = torch.empty(size=(heads*batch, N, d), dtype=torch.float32, device=device)
    
    stride_o_batch = o_tensor.stride(0)
    stride_do_batch = do_tensor.stride(0)
    stride_q_batch = q_tensor.stride(0)
    stride_k_batch = k_tensor.stride(0)
    stride_v_batch = v_tensor.stride(0)
    stride_L_batch = L_tensor.stride(0)
    stride_D_batch = D_tensor.stride(0)
    
    stride_o_row = o_tensor.stride(1)
    stride_do_row = do_tensor.stride(1)
    stride_q_row = q_tensor.stride(1)
    stride_k_col = k_tensor.stride(1)
    stride_v_col = v_tensor.stride(1)
    
    size_d = triton.next_power_of_2(d)
    
    grid_D = lambda META : (triton.cdiv(N, META["BS_row"]), batch*heads)
    
    triton_dtype = torch_to_triton_dtypes[dtype]
    
    args_D = (
        o_tensor, do_tensor,
        D_tensor,
        N, d, size_d,
        stride_o_batch,
        stride_do_batch, 
        stride_D_batch,
        stride_o_row,
        stride_do_row,
    )
    
    _kernel_D_fa2[grid_D](*args_D)      #type:ignore
    
    sqrt_d = math.sqrt(d)
    size_d = triton.next_power_of_2(d)
    
    grid_backward = lambda META : (triton.cdiv(N, META["BS_col"]), batch*heads)
    
    args_backward = (
        q_tensor, dq_tensor,
        k_tensor, dk_tensor,
        v_tensor, dv_tensor,
        do_tensor,
        L_tensor, D_tensor,
        stride_q_batch,
        stride_k_batch,
        stride_v_batch,
        stride_do_batch,
        stride_L_batch,
        stride_D_batch,
        stride_q_row,
        stride_k_col,
        stride_v_col,
        stride_do_row,
        N, d, sqrt_d, size_d,
        triton_dtype,
    )
    
    _kernel_fa2_backward[grid_backward](*args_backward) #type:ignore
    
    D_tensor = D_tensor.view(batch, heads, N).to(dtype=dtype)
    dq_tensor = dq_tensor.view(batch, heads, N, d).to(dtype=dtype)
    dk_tensor = dk_tensor.view(batch, heads, N, d).to(dtype=dtype)
    dv_tensor = dv_tensor.view(batch, heads, N, d).to(dtype=dtype)
    
    
    return D_tensor, dq_tensor, dk_tensor, dv_tensor


def _warmup_autograd():
    a = torch.randn(8, 8, device="cuda", requires_grad=True)
    b = torch.randn(8, 8, device="cuda", requires_grad=True)
    (a @ b).sum().backward()   # force l'init du contexte sur le thread autograd
    torch.cuda.synchronize()


if __name__ == "__main__":
    B = 32
    H = 56
    N = 56
    d = 128
    
    dtype = torch.bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    _warmup_autograd()
    
    q_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device, requires_grad=True)
    k_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device, requires_grad=True)
    v_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device, requires_grad=True)
    
    o_tensor, L_tensor = fa2_forward(q_tensor, k_tensor, v_tensor)
    o_torch = F.scaled_dot_product_attention(q_tensor, k_tensor, v_tensor)
    
    # torch.testing.assert_close(o_tensor, o_torch, atol=1e-2, rtol=1e-2)
    
    do_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device, requires_grad=True)
    
    D_tensor, dq_tensor, dk_tensor, dv_tensor = fa2_backward(q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, L_tensor)
    
    D_ref = ref_D_tensor(o_tensor, do_tensor, output_dtype=dtype)
    
    # print(f"The kernel tensor is equal to : \n{D_tensor}")
    # print(f"The ref tensor is equal to : \n{D_ref}")
    
    # print(f"The shape of the output D kernel is : \n{D_tensor.shape}")
    # print(f"The shape of the ref D is : \n{D_ref.shape}")
    
    torch.testing.assert_close(D_ref, D_tensor, atol=1e-2, rtol=1e-2)
    
    grad_q, grad_k, grad_v = torch.autograd.grad(
        outputs=o_torch,
        inputs=[q_tensor, k_tensor, v_tensor],
        grad_outputs=do_tensor
    )
    
    torch.testing.assert_close(grad_q, dq_tensor, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(grad_k, dk_tensor, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(grad_v, dv_tensor, atol=1e-2, rtol=1e-2)