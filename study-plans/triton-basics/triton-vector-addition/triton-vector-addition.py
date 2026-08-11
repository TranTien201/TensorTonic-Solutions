import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offset < n

    x_load = tl.load(x_ptr + offset, mask=mask)
    y_load = tl.load(y_ptr + offset, mask=mask)

    out = x_load + y_load
    
    tl.store(out_ptr + offset, out, mask=mask)


def solve(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch vector_add_kernel on the provided tensors."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)