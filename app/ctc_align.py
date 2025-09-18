from __future__ import annotations
import torch
from typing import List, Tuple

# emission: (T, V) の log-softmax 済みテンソル
# tokens: 参照列のトークンID配列（blankは別途指定）


def build_trellis(
    emission: torch.Tensor, tokens: List[int], blank_id: int
) -> torch.Tensor:
    T, V = emission.size()
    N = len(tokens)
    trellis = torch.empty((T, N + 1), device=emission.device).fill_(-float("inf"))
    trellis[0, 0] = 0.0
    for t in range(1, T):
        trellis[t, 0] = trellis[t - 1, 0] + emission[t, blank_id]
    for n in range(1, N + 1):
        trellis[0, n] = -float("inf")
    for t in range(1, T):
        for n in range(1, N + 1):
            stay = trellis[t - 1, n] + emission[t, blank_id]
            change = trellis[t - 1, n - 1] + emission[t, tokens[n - 1]]
            trellis[t, n] = torch.maximum(stay, change)
    return trellis


def backtrack(
    trellis: torch.Tensor, emission: torch.Tensor, tokens: List[int], blank_id: int
):
    t, n = trellis.size(0) - 1, trellis.size(1) - 1
    path = []
    while n > 0 and t > 0:
        p_stay = trellis[t - 1, n] + emission[t, blank_id]
        p_change = trellis[t - 1, n - 1] + emission[t, tokens[n - 1]]
        if p_change > p_stay:
            path.append((t, n, tokens[n - 1]))
            n -= 1
        else:
            path.append((t, n, blank_id))
        t -= 1
    path.reverse()
    return path


def merge_repeats(path, blank_id: int):
    merged = []
    if not path:
        return merged
    s_t, _, tok = path[0]
    prev_tok = tok
    for t, _, tok in path[1:]:
        if tok != prev_tok:
            if prev_tok != blank_id:
                merged.append((s_t, t, prev_tok))
            s_t = t
            prev_tok = tok
    if prev_tok != blank_id:
        merged.append((s_t, path[-1][0] + 1, prev_tok))
    return merged
