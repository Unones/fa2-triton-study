import torch

def ref_D_tensor(
    o_tensor : torch.Tensor,
    do_tensor : torch.Tensor,
    output_dtype : torch.dtype
):
    """
    A reference function in pytorch to calculate the tensor D
    used in the backward of flash attention 2.
       
    """
    
    o_tensor = o_tensor.to(dtype=torch.float32)
    do_tensor = do_tensor.to(dtype=torch.float32)
    
    pointwise_mul = o_tensor * do_tensor
    
    D_mat = torch.sum(pointwise_mul, dim=1)
    D_mat = D_mat.to(dtype=output_dtype)
    
    return D_mat
    
    
    