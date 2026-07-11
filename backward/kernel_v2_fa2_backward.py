import os
os.environ["TRITON_INTERPRET"] = "1"

import triton
import triton.language as tl
import torch


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
    
    Parameters
    ----------
    o_ptr
        Pointer to the output tensor o.
    do_ptr
        Pointer to the gradients of the output tensor do.
    D_ptr
        Pointer to the intermediate tensor D.
    size_n
        Another name for the dimension N.
    d
        Another name for the dimension d.
    size_d
        The next power of 2 of d.
    stride_o_batch
        The stride for the first dimension of the tensor o.
    stride_do_batch
        The stride for the first dimension of the tensor do.
    stride_D_batch
        The stride for the first dimension of the tensor D.
    stride_o_row
        The stride for the second dimension of the tensor o.
    stride_do_row
        The stride for the second dimension of the tensor do.
    BS_row
        The block size along the second dimension.
    
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
    
