import os
os.environ["TRITON_INTERPRET"] = "1"

import triton
import triton.language as tl
import torch
import torch.nn.functional as F

from forward.ref_fa2_forward import ref_fa2_forward

from math import ceil, sqrt

@triton.jit
def _kernel_fa2_forward(
    q_ptr, k_ptr, v_ptr, o_ptr, L_ptr,
    size_row, size_col, hidden_dimension : tl.constexpr, d,
    stride_q_row, stride_k_col,
    stride_v_col,
    output_dtype,
    nb_tiles_col : tl.constexpr,
    BS_row : tl.constexpr,
    BS_col : tl.constexpr,
    
):
    """
    Compute the attention mecanism using the Flash Attention 2 algorithm.
    
    """
    
    pid_row = tl.program_id(0)
    
    offset_row = pid_row*BS_row + tl.arange(0, BS_row)
    
    # tl.device_print("The offset_row is equal to : \n", offset_row)
    
    offset_q_row = offset_row * stride_q_row
    mask_row = offset_row < size_row
    
    # tl.device_print("The mask_row is equal to : \n", mask_row)
    
    offset_d = tl.arange(0, hidden_dimension)
    mask_d = offset_d < d
    
    # tl.device_print("The offset_d is equal to : \n", offset_d)
    
    offset_q = offset_q_row[:, None] + offset_d[None, :]
    mask_q = mask_row[:, None] & mask_d[None, :]
    
    tl.device_print("The offset_q is equal to : \n", offset_q)
    tl.device_print("The mask_q is equal to : \n", mask_q)
    
    q = tl.load(q_ptr + offset_q, mask=mask_q, other=0)
    
    o_row = tl.zeros((BS_row, hidden_dimension), dtype=tl.float32)
    l_row = tl.zeros((BS_row,), dtype=tl.float32)
    m_row = tl.full((BS_row,), float("-inf"), dtype=tl.float32)
    
    # tl.device_print("The tensor o_row is equal to \n", o_row)
    # tl.device_print("The tensor l_row is equal to \n", l_row)
    # tl.device_print("The tensor m_row is equal to \n", m_row)
    
    for j in range(nb_tiles_col):
        offset_col = j * BS_col + tl.arange(0, BS_col)
        mask_col = offset_col < size_col
        
        # tl.device_print("The offset_col is equal to \n", offset_col)
        # tl.device_print("The mask_col is equal to \n", mask_col)
        
        offset_k_col = offset_col * stride_k_col
        offset_v_col = offset_col * stride_v_col
        
        offset_k = offset_k_col[:, None] + offset_d[None, :]
        offset_v = offset_v_col[:, None] + offset_d[None, :]
        
        # tl.device_print("The offset_k is equal to \n", offset_k)
        # tl.device_print("The offset_v is equal to \n", offset_v)
        
        mask_kv = mask_col[:, None] & mask_d[None, :]
        
        k = tl.load(k_ptr + offset_k, mask=mask_kv, other=0).to(dtype=tl.float32)
        v = tl.load(v_ptr + offset_v, mask=mask_kv, other=0).to(dtype=tl.float32)
        
        mask_s = mask_row[:, None] & mask_col[None, :]
        
        former_m_row = m_row
        
        k_t = tl.trans(k)
        s = tl.dot(q, k_t)
        
        # tl.device_print("s is equal to : \n", s)
        
        max_row_s = tl.max(s, axis=1)
        m_row = tl.maximum(former_m_row, max_row_s)
        
        intermediate_matrix_p = tl.where(mask_s, s - m_row[:, None], float("-inf"))
        p = tl.exp(intermediate_matrix_p)
        
        # tl.device_print("max_row_s is equal to : \n", max_row_s)
        # tl.device_print("m_row is equal to \n", m_row)
        # tl.device_print("intermediate_matrix_p is equal to \n", intermediate_matrix_p)
        # tl.device_print("p is equal to \n", p)
        
        l_row_term_1 = tl.exp(former_m_row - m_row)
        l_row_term_2 = tl.sum(p, axis=1)
        l_row = l_row_term_1 * l_row + l_row_term_2
        
        tl.device_print("l_row is equal to : \n", l_row)
        tl.device_print("The first final o_row is equal to :\n", o_row)
        
        o_row_term_1 = o_row / (l_row_term_1[:, None] +1e-9)
        
        tl.device_print("The term1 of o_row is equal to :\n", o_row)
        
        o_row_term_2 = tl.dot(p, v)
        o_row = o_row_term_1 + o_row_term_2
        tl.device_print("The term2 of o_row is equal to :\n", o_row)
    
    # tl.device_print("The final l_row is equal o : \n", l_row)
    
    # tl.device_print("The first final o_row is equal to :\n", o_row)
    
    o_row = o_row / (l_row[:, None] +1e-9)
    
    tl.device_print("The second final o_row is equal to :\n", o_row)
    L_row = m_row + tl.log(l_row + 1e-9)
    
    o_row = o_row.to(dtype=output_dtype)
    L_row = L_row.to(dtype=output_dtype)    
    
    tl.store(o_ptr + offset_q, o_row, mask=mask_q)
    tl.store(L_ptr + offset_row, L_row, mask=mask_row)
    

def fa2_forward(
    q_tensor : torch.Tensor, 
    k_tensor : torch.Tensor, 
    v_tensor : torch.Tensor   
):
    """
    
    
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
    
    o_tensor = torch.empty((N, d), dtype=dtype, device=device)
    L_tensor = torch.empty((N,), dtype=dtype, device=device)
    
    BS_row = 4
    BS_col = 4
    nb_tiles_row = ceil(N / BS_row)
    nb_tiles_col = ceil(N / BS_col)
    
    hidden_dimension = triton.next_power_of_2(d)
    
    grid = (nb_tiles_row, )
    
    args = (
        q_tensor, k_tensor, v_tensor,
        o_tensor, L_tensor, 
        N, N, hidden_dimension, d,
        stride_q_row, stride_k_row, 
        stride_v_row, 
        tl.float32,
        nb_tiles_col, BS_row, BS_col
    )
    
    _kernel_fa2_forward[grid](*args)        #type:ignore
    
    return o_tensor, L_tensor
    

if __name__ == "__main__":
    N = 4
    d = 4
    
    dtype = torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    q_tensor = torch.randn((N, d), dtype=dtype, device=device)
    k_tensor = torch.randn((N, d), dtype=dtype, device=device)
    v_tensor = torch.randn((N, d), dtype=dtype, device=device)
    
    o_tensor, L_tensor = fa2_forward(q_tensor, k_tensor, v_tensor)
    
    o_torch = F.scaled_dot_product_attention(q_tensor, k_tensor, v_tensor)
    
    s_ref, p_ref, o_ref = ref_fa2_forward(q_tensor, k_tensor, v_tensor)
    
    # torch.testing.assert_close(o_torch, o_ref, atol=1e-3, rtol=1e-3)
    
    # torch.testing.assert_close(o)
    
    # print(p_ref)
    
    print(o_torch)
    # print(o_tensor)
    print(o_ref)
    
    
    