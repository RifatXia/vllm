# SPDX-License-Identifier: Apache-2.0
"""KVLobotomy pre-RoPE experiment support.

When KVLOBOTOMY_PRE_ROPE=1 is set in the environment (before importing vllm),
the RoPE application order changes:
  - Post-RoPE (default): RoPE applied to Q and K before KV cache storage
  - Pre-RoPE (this flag): RoPE applied to Q only; K stored unrotated;
    RoPE applied to K at attention time
"""
import os
import threading

import torch

KVLOBOTOMY_PRE_ROPE = os.environ.get("KVLOBOTOMY_PRE_ROPE", "0") == "1"

_local = threading.local()


def set_rope_context(positions, cos_sin_cache, rotary_dim, head_size,
                     is_neox_style):
    """Store RoPE context so attention can apply deferred RoPE to K."""
    _local.positions = positions
    _local.cos_sin_cache = cos_sin_cache
    _local.rotary_dim = rotary_dim
    _local.head_size = head_size
    _local.is_neox_style = is_neox_style


def get_rope_context():
    return (
        getattr(_local, 'positions', None),
        getattr(_local, 'cos_sin_cache', None),
        getattr(_local, 'rotary_dim', None),
        getattr(_local, 'head_size', None),
        getattr(_local, 'is_neox_style', None),
    )


def apply_rope_to_key(key):
    """Apply deferred RoPE rotation to key tensor.

    Args:
        key: [num_tokens, num_kv_heads, head_size]
    Returns:
        key with RoPE applied (new tensor)
    """
    from vllm.model_executor.layers.rotary_embedding.common import (
        ApplyRotaryEmb,
    )

    positions, cos_sin_cache, rotary_dim, _, is_neox_style = (
        get_rope_context())
    if positions is None or cos_sin_cache is None:
        return key

    pos = positions.flatten()
    # Only rotate up to the number of key tokens
    if pos.shape[0] > key.shape[0]:
        pos = pos[:key.shape[0]]

    cos_sin = cos_sin_cache.index_select(0, pos)
    cos, sin = cos_sin.chunk(2, dim=-1)

    k_rot = key[..., :rotary_dim]
    k_pass = key[..., rotary_dim:]
    k_rot = ApplyRotaryEmb.forward_static(k_rot, cos, sin, is_neox_style)
    return torch.cat((k_rot, k_pass), dim=-1)
