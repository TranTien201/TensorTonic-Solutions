import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n

    # other=0.0 trong tl.load: khi mask=False (phần tử ngoài biên n), nếu không set other, giá trị load về là "garbage" (rác trong bộ nhớ) — cộng rác vào tổng sẽ ra kết quả sai. other=0.0 đảm bảo phần tử ngoài biên đóng góp 0 vào sum.
    x_load = tl.load(x_ptr + offset, mask=mask, other=0.0)

    out = tl.sum(x_load, axis=0)
    tl.atomic_add(out_ptr, out)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch sum_kernel on the provided tensors."""
    n = x.numel()
    out.zero_()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    sum_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)