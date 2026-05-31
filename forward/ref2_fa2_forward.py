import torch

from math import ceil, sqrt

def ref2_fa2_forward(
    q_tensor : torch.Tensor, 
    k_tensor : torch.Tensor, 
    v_tensor : torch.Tensor
):
    """
    
    """
    
    BS_row = 4
    BS_col = 4
    
    N, d = q_tensor.shape
    dtype = q_tensor.dtype
    device = q_tensor.device
    
    o_tensor = torch.empty((N, d), dtype=dtype, device=device)
    L_tensor = torch.empty((N,), dtype=dtype, device=device)
    
    nb_tiles_row = ceil(N / BS_row)
    nb_tiles_col = ceil(N / BS_col)
    
    for i in range(1, nb_tiles_row+1):
        q = q_tensor[(i-1)*BS_row : i*BS_row]
        
        o_row = torch.zeros((BS_row, d), dtype=dtype, device=device)
        l_row = torch.zeros((BS_row,), dtype=dtype, device=device)
        m_row = torch.full((BS_row,), fill_value=-float("inf"), dtype=dtype, device=device)

        for j in range(1, nb_tiles_col+1):
            k = k_tensor[(j-1)*BS_col : j*BS_col]
            v = v_tensor[(j-1)*BS_col : j*BS_col]

            k_t = torch.transpose(k, 0, 1)
            s = q @ k_t / sqrt(d)
            
            rowmax_s = torch.max(s, dim=1)
            former_m_row = m_row
            m_row = torch.maximum(m_row,rowmax_s.values)
            
            p = torch.exp(s - m_row[:, None])
            
            l_row_term_1 = torch.exp(former_m_row - m_row)
            l_row_term_2 = torch.sum(p, dim=1)
            l_row = l_row_term_1 * l_row + l_row_term_2
            
            o_row = o_row * l_row_term_1[:, None] + p@v

            
        o_row = o_row / l_row[:, None]
        L_row = m_row + torch.log(l_row)
        
        o_tensor[(i-1)*BS_row : i*BS_row] = o_row
        L_tensor[(i-1)*BS_row : i*BS_row] = L_row
    
    return o_tensor, L_tensor



if __name__ == "__main__":
    N = 8
    d = 4
    
    dtype = torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    q_tensor = torch.randn((N, d), dtype=dtype, device=device)
    k_tensor = torch.randn((N, d), dtype=dtype, device=device)
    v_tensor = torch.randn((N, d), dtype=dtype, device=device)
    # print(k_tensor)
    
    o_ref2, _ = ref2_fa2_forward(q_tensor, k_tensor, v_tensor)
    
    print(o_ref2)
    
    
    
            