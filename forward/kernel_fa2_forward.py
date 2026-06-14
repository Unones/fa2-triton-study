import triton
import triton.language as tl
import torch
import torch.nn.functional as F

from math import ceil, sqrt

@triton.jit
def _kernel_fa2_forward(
    q_ptr, k_ptr, v_ptr, o_ptr, L_ptr,
    size_row, size_col, hidden_dimension : tl.constexpr, d, d_sqrt,
    stride_q_row, stride_k_col,
    stride_v_col,
    output_dtype : tl.constexpr,
    nb_tiles_col : tl.constexpr,
    BS_row : tl.constexpr,
    BS_col : tl.constexpr,
    
):
    """
    Compute the attention mecanism using the Flash Attention 2 algorithm.
    
    """
    
    pid_row = tl.program_id(0)
    
    offset_row = pid_row*BS_row + tl.arange(0, BS_row)
    offset_q_row = offset_row * stride_q_row
    mask_row = offset_row < size_row

    offset_d = tl.arange(0, hidden_dimension)
    mask_d = offset_d < d

    offset_q = offset_q_row[:, None] + offset_d[None, :]
    mask_q = mask_row[:, None] & mask_d[None, :]

    q = tl.load(q_ptr + offset_q, mask=mask_q, other=0).to(dtype=tl.float32)
    
    o_row = tl.zeros((BS_row, hidden_dimension), dtype=tl.float32)
    l_row = tl.zeros((BS_row,), dtype=tl.float32)
    m_row = tl.full((BS_row,), float("-inf"), dtype=tl.float32)

    for j in range(nb_tiles_col):
        offset_col = j * BS_col + tl.arange(0, BS_col)
        mask_col = offset_col < size_col

        offset_k_col = offset_col * stride_k_col
        offset_v_col = offset_col * stride_v_col
        
        offset_k = offset_k_col[:, None] + offset_d[None, :]
        offset_v = offset_v_col[:, None] + offset_d[None, :]

        mask_kv = mask_col[:, None] & mask_d[None, :]
        
        k = tl.load(k_ptr + offset_k, mask=mask_kv, other=0).to(dtype=tl.float32)
        v = tl.load(v_ptr + offset_v, mask=mask_kv, other=0).to(dtype=tl.float32)
        
        mask_s = mask_row[:, None] & mask_col[None, :]
        
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

        o_row_term_2 = tl.dot(p, v)
        o_row = o_row_term_1 + o_row_term_2

    o_row = o_row / (l_row[:, None] + 1e-6)

    L_row = m_row + tl.log(l_row + 1e-9)
    
    o_row = o_row.to(dtype=output_dtype)
    
    tl.store(o_ptr + offset_q, o_row, mask=mask_q)
    tl.store(L_ptr + offset_row, L_row, mask=mask_row)
    
torch_to_triton_dtypes = {
    torch.float32 : tl.float32,
    torch.float16 : tl.float16,
    torch.bfloat16 : tl.bfloat16
}


def fa2_forward(
    q_tensor : torch.Tensor, 
    k_tensor : torch.Tensor, 
    v_tensor : torch.Tensor
):
    """
    Computes the forward attention mechanism using flash attention 2.
    
    """
    
    q_tensor = q_tensor.contiguous()
    k_tensor = k_tensor.contiguous()
    v_tensor = v_tensor.contiguous()
    
    dtype=q_tensor.dtype
    device=q_tensor.device
    
    stride_q_row = q_tensor.stride(0)
    stride_k_row = k_tensor.stride(0)
    stride_v_row = v_tensor.stride(0)
    
    N, d = q_tensor.shape
    d_sqrt = sqrt(d)
    
    o_tensor = torch.empty((N, d), dtype=dtype, device=device)
    L_tensor = torch.empty((N,), dtype=torch.float32, device=device)
    
    BS_row = 64
    BS_col = 64
    nb_tiles_row = ceil(N / BS_row)
    nb_tiles_col = ceil(N / BS_col)
    
    hidden_dimension = triton.next_power_of_2(d)
    
    grid = (nb_tiles_row, )
    
    args = (
        q_tensor, k_tensor, v_tensor,
        o_tensor, L_tensor, 
        N, N, hidden_dimension, d, d_sqrt,
        stride_q_row, stride_k_row, 
        stride_v_row, 
        torch_to_triton_dtypes[dtype],
        nb_tiles_col, BS_row, BS_col
    )
    
    _kernel_fa2_forward[grid](*args)        #type:ignore
    
    return o_tensor, L_tensor
    

if __name__ == "__main__":
    N = 128
    d = 64
    
    dtype = torch.bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    q_tensor = torch.randn((N, d), dtype=dtype, device=device)
    k_tensor = torch.randn((N, d), dtype=dtype, device=device)
    v_tensor = torch.randn((N, d), dtype=dtype, device=device)
    
    o_tensor, L_tensor = fa2_forward(q_tensor, k_tensor, v_tensor)
    
    o_torch = F.scaled_dot_product_attention(q_tensor, k_tensor, v_tensor)

    torch.testing.assert_close(o_torch, o_tensor, atol=1e-2, rtol=1e-2)
    