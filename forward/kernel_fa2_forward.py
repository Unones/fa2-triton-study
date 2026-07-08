# import os
# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import triton
import triton.language as tl
import torch
import torch.nn.functional as F

from math import ceil, sqrt

@triton.autotune(configs=[
    triton.Config(kwargs={"BS_row" : 16, "BS_col" : 16}, num_stages=3),
    # triton.Config(kwargs={"BS_row" : 16, "BS_col" : 16}, num_stages=4),
    # triton.Config(kwargs={"BS_row" : 16, "BS_col" : 16}, num_stages=5),
    # triton.Config(kwargs={"BS_row" : 32, "BS_col" : 32}, num_stages=2),
    # triton.Config(kwargs={"BS_row" : 32, "BS_col" : 32}, num_stages=3),
    # triton.Config(kwargs={"BS_row" : 32, "BS_col" : 32}, num_stages=4),
    # triton.Config(kwargs={"BS_row" : 32, "BS_col" : 64}, num_stages=3),
    # triton.Config(kwargs={"BS_row" : 32, "BS_col" : 64}, num_stages=4),
    # triton.Config(kwargs={"BS_row" : 64, "BS_col" : 64}, num_stages=2),
    # triton.Config(kwargs={"BS_row" : 64, "BS_col" : 64}, num_stages=3),
    # triton.Config(kwargs={"BS_row" : 64, "BS_col" : 64}, num_stages=4),
    # triton.Config(kwargs={"BS_row" : 16, "BS_col" : 128}, num_stages=2),
    # triton.Config(kwargs={"BS_row" : 16, "BS_col" : 128}, num_stages=3),

],
    key=["size_row", "size_col"]
)
@triton.jit
def _kernel_fa2_forward(
    q_ptr, k_ptr, v_ptr, o_ptr, L_ptr,
    size_row, size_col, hidden_dimension : tl.constexpr, d, d_sqrt,
    batch_dim,
    stride_q_batch,
    stride_k_batch,
    stride_v_batch,
    stride_L_batch,
    stride_q_row, 
    stride_k_col,
    stride_v_col,
    output_dtype : tl.constexpr,
    BS_row : tl.constexpr,
    BS_col : tl.constexpr,
    
):
    """
    Compute the attention mecanism using the Flash Attention 2 algorithm.
    
    The first four pointers are of dimensions (H*B, N, d).
    
    Parameters
    ----------
    q_ptr
        The pointer to the query tensor.
    k_ptr
        The pointer to the key tensor.
    v_ptr
        The pointer to the value tensor.
    o_ptr
        The pointer to the output tensor.
    L_ptr
        The pointer to the intermediate tensor.
    size_row
        Another name for the dimension N.
    size_col
        Another name for the dimension N.
    hidden_dimension
        The dimension d put to the next power of 2.
    d
        The dimension d.
    d_sqrt
        Equal to sqrt(d).
    batch_dim
        The number H*B to the next power of 2.
    stride_q_batch
        The stride of the first dimension of q_ptr.
    stride_k_batch
        The stride of the first dimension of k_ptr.
    stride_v_batch
        The stride of the first dimension of v_ptr.
    stride_q_row
        The stride of the second dimension of q_ptr.
    stride_k_row
        The stride of the second dimension of k_ptr.
    stride_v_row
        The stride of the second dimension of v_ptr.
    output_dtype
        The dtype fo the output.
    BS_row
        The block size for the rows.
    BS_col
        The block size for the cols.
    
    """
    
    BS_batch = 1
    
    pid_row = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    offset_row = pid_row*BS_row + tl.arange(0, BS_row)
    offset_batch = pid_batch * BS_batch 
    offset_d = tl.arange(0, hidden_dimension)
    
    mask_row = offset_row < size_row
    mask_batch = offset_batch < batch_dim
    mask_d = offset_d < d
    
    offset_q_row = offset_row * stride_q_row
    offset_q_batch = offset_batch * stride_q_batch
    offset_q = offset_q_row[:, None] + offset_d[None, :] + offset_q_batch
    
    offset_L_batch = offset_batch * stride_L_batch
    offset_L = offset_row + offset_L_batch 

    mask_q = mask_row[:, None] & mask_d[None, :] & mask_batch
    mask_L = mask_row & mask_batch

    q = tl.load(q_ptr + offset_q, mask=mask_q, other=0)
    
    o_row = tl.zeros((BS_row, hidden_dimension), dtype=tl.float32)
    l_row = tl.zeros((BS_row,), dtype=tl.float32)
    m_row = tl.full((BS_row,), float("-inf"), dtype=tl.float32)
    
    nb_tiles_col = tl.cdiv(size_col, BS_col)

    for j in range(nb_tiles_col):
        offset_col = j * BS_col + tl.arange(0, BS_col)
        mask_col = offset_col < size_col

        offset_k_col = offset_col * stride_k_col
        offset_v_col = offset_col * stride_v_col
        
        offset_k_batch = offset_batch * stride_k_batch
        offset_v_batch = offset_batch * stride_v_batch
        
        offset_k = offset_k_col[:, None] + offset_d[None, :] + offset_k_batch
        offset_v = offset_v_col[:, None] + offset_d[None, :] + offset_v_batch

        mask_kv = mask_col[:, None] & mask_d[None, :] & mask_batch
        
        k = tl.load(k_ptr + offset_k, mask=mask_kv, other=0)
        v = tl.load(v_ptr + offset_v, mask=mask_kv, other=0)
        
        mask_s = mask_row[:, None] & mask_col[None, :] & mask_batch
        
        former_m_row = m_row
        
        k_t = tl.trans(k)
        s = tl.dot(q, k_t) / d_sqrt

        max_row_s = tl.max(s, axis=1)
        m_row = tl.maximum(former_m_row, max_row_s)
        
        intermediate_matrix_p = tl.where(mask_s, s - m_row[:, None], float("-inf"))
        p = tl.exp(intermediate_matrix_p)

        l_row_term_1 = tl.exp(former_m_row - m_row)
        l_row_term_2 = tl.sum(p, axis=1)
        l_row = l_row_term_1 * l_row + l_row_term_2

        o_row_term_1 = o_row * l_row_term_1[:, None]

        o_row_term_2 = tl.dot(p.to(dtype=output_dtype), v)
        o_row = o_row_term_1 + o_row_term_2

    o_row = o_row / (l_row[:, None] + 1e-6)

    L_row = m_row + tl.log(l_row + 1e-9)
    
    o_row = o_row.to(dtype=output_dtype)

    tl.store(o_ptr + offset_q, o_row, mask=mask_q)
    tl.store(L_ptr + offset_L, L_row, mask=mask_L)
    
torch_to_triton_dtypes = {
    torch.float32 : tl.float32,
    torch.float16 : tl.float16,
    torch.bfloat16 : tl.bfloat16
}


def fa2_forward(
    q_tensor : torch.Tensor, 
    k_tensor : torch.Tensor, 
    v_tensor : torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the forward attention mechanism using flash attention 2.
    
    All input tensors must be 4-dimensional : (H, B, N, d).
    
    Parameters
    ----------
    q_tensor
        The query tensor.
    k_tensor
        The key tensor.
    v_tensor
        The value tensor
    
    Returns
    -------
    o_tensor
        The output tensor
    L_tensor
        The intermediate tensor for the backward pass.

    
    """
    
    batch, heads, N, d = q_tensor.size()
    d_sqrt = sqrt(d)
    
    q_tensor = q_tensor.contiguous()
    k_tensor = k_tensor.contiguous()
    v_tensor = v_tensor.contiguous()
    
    dtype=q_tensor.dtype
    device=q_tensor.device
    
    q_tensor = q_tensor.view(heads*batch, N, d)
    k_tensor = k_tensor.view(heads*batch, N, d)
    v_tensor = v_tensor.view(heads*batch, N, d)
    
    stride_q_batch = q_tensor.stride(0)
    stride_k_batch = k_tensor.stride(0)
    stride_v_batch = v_tensor.stride(0)
    
    stride_q_row = q_tensor.stride(1)
    stride_k_row = k_tensor.stride(1)
    stride_v_row = v_tensor.stride(1)

    o_tensor = torch.empty((heads*batch, N, d), dtype=dtype, device=device)
    L_tensor = torch.empty((heads*batch, N,), dtype=torch.float32, device=device)
    
    stride_L_batch = L_tensor.stride(0)

    hidden_dimension = triton.next_power_of_2(d)
    
    grid = lambda META : (ceil(N / META["BS_row"]), heads*batch)
    
    args = (
        q_tensor, k_tensor, v_tensor,
        o_tensor, L_tensor, 
        N, N, hidden_dimension, d, d_sqrt,
        heads*batch,
        stride_q_batch,
        stride_k_batch,
        stride_v_batch,
        stride_L_batch,
        stride_q_row, 
        stride_k_row, 
        stride_v_row, 
        torch_to_triton_dtypes[dtype]
    )
    
    _kernel_fa2_forward[grid](*args)        #type:ignore
    
    o_tensor = o_tensor.view(batch, heads, N, d)
    L_tensor = L_tensor.view(batch, heads, N)
    
    return o_tensor, L_tensor
    

if __name__ == "__main__":
    H = 2
    B = 10
    N = 30
    d = 16
    
    torch.manual_seed(42)
    
    dtype = torch.bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    q_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device)
    k_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device)
    v_tensor = torch.randn((B, H, N, d), dtype=dtype, device=device)
    
    o_tensor, L_tensor = fa2_forward(q_tensor, k_tensor, v_tensor)
    
    o_torch = F.scaled_dot_product_attention(q_tensor, k_tensor, v_tensor)

    torch.testing.assert_close(o_torch, o_tensor, atol=1e-2, rtol=1e-2)
    