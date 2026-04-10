# SPDX-License-Identifier: Apache-2.0
"""KV cache deletion: remove a segment and compact remaining tokens.

Dispatched to the GPU worker via LLM.collective_rpc().

Usage:
    results = llm.collective_rpc(
        kvlobotomy_full_delete,
        args=(delete_start, delete_end, rope_theta, head_dim,
              num_kv_heads, total_seq_len),
    )
"""

import time

import torch


def apply_delta_rotation(
    keys: torch.Tensor,
    positions: torch.Tensor,
    delta: int,
    head_dim: int,
    rope_theta: float = 500000.0,
) -> torch.Tensor:
    """Apply a delta RoPE rotation to key vectors.

    Uses the group property: R(new_pos) = R(delta) * R(old_pos)
    So K_corrected = R(delta) * K_old.

    Args:
        keys: [N, num_kv_heads, head_dim] tensor of rotated keys
        positions: [N] tensor of original positions (unused for delta, kept
                   for API consistency)
        delta: position shift (typically negative, e.g., -len(B))
        head_dim: dimension of each attention head
        rope_theta: RoPE base frequency
    Returns:
        Corrected keys tensor [N, num_kv_heads, head_dim]
    """
    device = keys.device
    dtype = keys.dtype
    num_tokens, num_heads, d = keys.shape

    # Compute rotation frequencies
    freq_indices = torch.arange(0, d // 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (rope_theta ** (2.0 * freq_indices / d))

    # Delta angles
    angles = delta * freqs  # [d//2]
    cos_delta = torch.cos(angles).to(dtype)  # [d//2]
    sin_delta = torch.sin(angles).to(dtype)  # [d//2]

    # Split keys into pairs for rotation (neox style: first half, second half)
    k_even = keys[..., : d // 2]  # [N, H, d//2]
    k_odd = keys[..., d // 2 :]  # [N, H, d//2]

    # Apply 2D rotation to each pair
    new_even = k_even * cos_delta - k_odd * sin_delta
    new_odd = k_even * sin_delta + k_odd * cos_delta

    return torch.cat([new_even, new_odd], dim=-1)


def kvlobotomy_full_delete(worker, delete_start, delete_end, rope_theta,
                           head_dim, num_kv_heads, total_seq_len):
    """Remove a segment from the KV cache and compact remaining tokens.

    1. Read C+suffix keys/values from their current positions
    2. Delta-rotate C+suffix keys by -len(B) positions
    3. Write C+suffix to compacted positions (filling B's gap)
    4. Zero out the stale tail (old positions that are now unused)

    Args:
        worker: vLLM worker object (provides access to kv_caches)
        delete_start: start position of segment to delete (inclusive)
        delete_end: end position of segment to delete (exclusive)
        rope_theta: RoPE base frequency
        head_dim: attention head dimension
        num_kv_heads: number of KV heads
        total_seq_len: total sequence length before deletion

    Returns:
        dict with deletion_latency_ms, rotation_ms, copy_ms, new_seq_len,
        and metadata
    """
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    delete_len = delete_end - delete_start
    tokens_after = total_seq_len - delete_end  # C+suffix tokens to move
    new_seq_len = total_seq_len - delete_len

    rotation_ms = 0.0
    copy_ms = 0.0

    kv_caches = worker.model_runner.kv_caches

    for layer_idx, kv_cache in enumerate(kv_caches):
        key_cache = kv_cache[0]   # [num_blocks, block_size, kv_heads, head_dim]
        val_cache = kv_cache[1]   # [num_blocks, block_size, kv_heads, head_dim]
        block_size = key_cache.shape[1]

        torch.cuda.synchronize()
        t_copy_start = time.perf_counter()

        if tokens_after > 0:
            # Source positions (where C+suffix currently are)
            old_positions = torch.arange(
                delete_end, total_seq_len,
                device=key_cache.device, dtype=torch.long,
            )
            # Destination positions (where they go after compaction)
            new_positions = old_positions - delete_len

            # Map to block addresses
            old_blocks = old_positions // block_size
            old_offsets = old_positions % block_size
            new_blocks = new_positions // block_size
            new_offsets = new_positions % block_size

            # Gather C+suffix keys and values (fancy indexing = copy)
            c_keys = key_cache[old_blocks, old_offsets]
            c_vals = val_cache[old_blocks, old_offsets]

            # Delta-rotate keys to correct positions
            torch.cuda.synchronize()
            t_rot_start = time.perf_counter()

            c_keys = apply_delta_rotation(
                c_keys, old_positions, -delete_len, head_dim, rope_theta,
            )

            torch.cuda.synchronize()
            t_rot_end = time.perf_counter()
            rotation_ms += (t_rot_end - t_rot_start) * 1000.0

            # Scatter to compacted positions
            key_cache[new_blocks, new_offsets] = c_keys
            val_cache[new_blocks, new_offsets] = c_vals

        # Zero out stale tail: positions [new_seq_len, total_seq_len)
        # Prevents any residual data from being accessible
        stale_positions = torch.arange(
            new_seq_len, total_seq_len,
            device=key_cache.device, dtype=torch.long,
        )
        if len(stale_positions) > 0:
            stale_blocks = stale_positions // block_size
            stale_offsets = stale_positions % block_size
            key_cache[stale_blocks, stale_offsets] = 0
            val_cache[stale_blocks, stale_offsets] = 0

        torch.cuda.synchronize()
        t_copy_end = time.perf_counter()
        copy_ms += (t_copy_end - t_copy_start) * 1000.0

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    deletion_latency_ms = (end_time - start_time) * 1000.0

    # copy_ms includes rotation_ms; separate them
    copy_only_ms = copy_ms - rotation_ms

    return {
        "deletion_latency_ms": deletion_latency_ms,
        "rotation_ms": rotation_ms,
        "copy_ms": copy_only_ms,
        "total_tokens_moved": tokens_after,
        "total_tokens_deleted": delete_len,
        "new_seq_len": new_seq_len,
        "delete_start": delete_start,
        "delete_end": delete_end,
    }
