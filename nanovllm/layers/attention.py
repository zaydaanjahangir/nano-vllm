import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Pure-PyTorch (SDPA) replacement for flash-attn's varlen + paged-KV
        # kernels. Slower, but has no compiled-extension dependency, and
        # makes the paged-KV gather explicit instead of hiding it in a kernel.
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        n_rep = self.num_heads // self.num_kv_heads

        if context.is_prefill:
            if (
                context.block_tables is not None
            ):  # prefix cache: gather full K/V from paged cache
                k, v = self._gather_prefill_kv(context)
            o = self._varlen_causal_sdpa(q, k, v, context, n_rep)
        else:  # decode: gather each sequence's cached K/V via its block table
            o = self._decode_sdpa(q, k_cache, v_cache, context, n_rep)
        return o

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        # x: [seq, num_kv_heads, head_dim] -> [seq, num_kv_heads * n_rep, head_dim]
        if n_rep == 1:
            return x
        s, h, d = x.shape
        return x[:, :, None, :].expand(s, h, n_rep, d).reshape(s, h * n_rep, d)

    def _varlen_causal_sdpa(self, q, k, v, context, n_rep):
        cu_q = context.cu_seqlens_q.tolist()
        cu_k = context.cu_seqlens_k.tolist()
        outs = []
        for i in range(len(cu_q) - 1):
            qi = q[cu_q[i] : cu_q[i + 1]]  # [seqlen_q, num_heads, head_dim]
            ki = k[cu_k[i] : cu_k[i + 1]]  # [seqlen_k, num_kv_heads, head_dim]
            vi = v[cu_k[i] : cu_k[i + 1]]
            ki = self._repeat_kv(ki, n_rep)
            vi = self._repeat_kv(vi, n_rep)
            # SDPA wants [batch, heads, seq, dim]
            qi = qi.transpose(0, 1).unsqueeze(0)  # [1, H, Sq, D]
            ki = ki.transpose(0, 1).unsqueeze(0)  # [1, H, Sk, D]
            vi = vi.transpose(0, 1).unsqueeze(0)
            seqlen_q, seqlen_k = qi.size(2), ki.size(2)
            if seqlen_q == seqlen_k:
                # no prefix cache: standard causal mask
                oi = F.scaled_dot_product_attention(
                    qi, ki, vi, scale=self.scale, is_causal=True
                )
            else:
                # prefix cache: query tokens sit at the END of the key sequence.
                # Row j (query) may attend to keys [0, seqlen_k - seqlen_q + j].
                offset = seqlen_k - seqlen_q
                q_idx = torch.arange(seqlen_q, device=qi.device).unsqueeze(1)
                k_idx = torch.arange(seqlen_k, device=qi.device).unsqueeze(0)
                attn_mask = k_idx <= (q_idx + offset)
                oi = F.scaled_dot_product_attention(
                    qi, ki, vi, attn_mask=attn_mask, scale=self.scale
                )
            outs.append(oi.squeeze(0).transpose(0, 1))  # [Sq, H, D]
        return torch.cat(outs, dim=0)

    def _gather_prefill_kv(self, context):
        # Reconstruct full dense K/V (per sequence, concatenated) from the
        # paged cache using block_tables, for sequences with a cache hit.
        block_tables = context.block_tables
        block_size = self.k_cache.size(1)
        cu_k = context.cu_seqlens_k.tolist()
        ks, vs = [], []
        for i, bt in enumerate(block_tables.tolist()):
            seqlen_k = cu_k[i + 1] - cu_k[i]
            blocks = [b for b in bt if b != -1]
            k_full = self.k_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[
                :seqlen_k
            ]
            v_full = self.v_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[
                :seqlen_k
            ]
            ks.append(k_full)
            vs.append(v_full)
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0)

    def _decode_sdpa(self, q, k_cache, v_cache, context, n_rep):
        # q: [num_seqs, num_heads, head_dim] (one new token per sequence)
        block_tables = context.block_tables.tolist()
        context_lens = context.context_lens.tolist()
        outs = []
        for i, (bt, clen) in enumerate(zip(block_tables, context_lens)):
            if clen == 0:  # CUDA-graph warmup/capture: no tokens cached yet
                outs.append(torch.zeros_like(q[i]))
                continue
            blocks = [b for b in bt if b != -1]
            ki = k_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:clen]
            vi = v_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:clen]
            ki = (
                self._repeat_kv(ki, n_rep).transpose(0, 1).unsqueeze(0)
            )  # [1, H, clen, D]
            vi = self._repeat_kv(vi, n_rep).transpose(0, 1).unsqueeze(0)
            qi = q[i].unsqueeze(0).unsqueeze(2)  # [1, H, 1, D]
            oi = F.scaled_dot_product_attention(qi, ki, vi, scale=self.scale)
            outs.append(oi.squeeze(0).squeeze(1))  # [H, D]
        return torch.stack(outs, dim=0)
