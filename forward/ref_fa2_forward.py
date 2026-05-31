import torch
import torch.nn.functional as F

from math import sqrt

def ref_fa2_forward(
    q_tensor : torch.Tensor, 
    k_tensor : torch.Tensor, 
    v_tensor : torch.Tensor 
):
    """
    A reference function of the implementation of FA2.
    
    """
    _, d = q_tensor.shape
    
    S = q_tensor @ torch.transpose(k_tensor, 0, 1) 
    P = F.softmax(S, dim=1)
    O = P @ v_tensor
    
    return S, P, O
    
    