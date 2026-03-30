# SPDX-License-Identifier: Apache-2.0
"""KVLobotomy operations: segment deletion with RoPE correction.

This module implements the delete() operation for both pre-RoPE and post-RoPE
storage modes. It is designed to be dispatched to the GPU worker via
LLM.collective_rpc().

Usage from experiment script:
    import os
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    results = llm.collective_rpc(
        kvlobotomy_delete,
        args=(delete_start, delete_end, rope_storage, rope_theta, head_dim,
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


def apply_rope_rotation(
    keys: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    rope_theta: float = 500000.0,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply or undo RoPE rotation on key vectors.

    Args:
        keys: [N, num_kv_heads, head_dim] tensor
        positions: [N] tensor of position indices
        head_dim: dimension of each attention head
        rope_theta: RoPE base frequency
        inverse: if True, undo the rotation (apply R(-pos))
    Returns:
        Rotated (or unrotated) keys tensor [N, num_kv_heads, head_dim]
    """
    device = keys.device
    dtype = keys.dtype
    d = head_dim

    freq_indices = torch.arange(0, d // 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (rope_theta ** (2.0 * freq_indices / d))  # [d//2]

    # angles: [N, d//2] — per-token, per-dimension
    angles = positions.float().unsqueeze(1) * freqs.unsqueeze(0)  # [N, d//2]
    if inverse:
        angles = -angles

    cos_vals = torch.cos(angles).to(dtype)  # [N, d//2]
    sin_vals = torch.sin(angles).to(dtype)  # [N, d//2]

    # Expand for heads: [N, 1, d//2] broadcasts over num_heads
    cos_vals = cos_vals.unsqueeze(1)
    sin_vals = sin_vals.unsqueeze(1)

    # Neox-style rotation
    k_even = keys[..., : d // 2]
    k_odd = keys[..., d // 2 :]

    new_even = k_even * cos_vals - k_odd * sin_vals
    new_odd = k_even * sin_vals + k_odd * cos_vals

    return torch.cat([new_even, new_odd], dim=-1)


def kvlobotomy_delete(worker, delete_start, delete_end, rope_storage,
                      rope_theta, head_dim, num_kv_heads, total_seq_len):
    """Simulate deletion overhead by undoing and redoing RoPE on ALL cached keys.

    For post-RoPE: performs undo+redo RoPE on every key (A+B+C) across all
    layers. This measures the computational cost of re-encoding positions
    for the entire cache. The cache is NOT modified — original keys are
    written back after the computation.

    For pre-RoPE: no-op. Keys are stored unrotated, so there is no position
    encoding work to simulate.

    This function runs on the GPU worker process via collective_rpc.

    Args:
        worker: vLLM worker object (provides access to kv_caches)
        delete_start: start position of segment B (inclusive) — used for
                      metadata reporting only
        delete_end: end position of segment B (exclusive) — used for
                    metadata reporting only
        rope_storage: "pre" or "post"
        rope_theta: RoPE base frequency (e.g., 500000.0 for Llama 3.2)
        head_dim: attention head dimension
        num_kv_heads: number of KV heads
        total_seq_len: total sequence length in cache

    Returns:
        dict with deletion_latency_ms and metadata
    """
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    if rope_storage == "pre":
        # Pre-RoPE: keys are stored unrotated. No position encoding
        # work needed — cost is O(1).
        pass

    elif rope_storage == "post":
        # Post-RoPE: simulate the cost of undoing and redoing RoPE
        # on ALL keys in the cache across ALL layers.
        # The cache is NOT modified — originals are written back.
        if total_seq_len > 0:
            kv_caches = worker.model_runner.kv_caches
            for layer_idx, kv_cache in enumerate(kv_caches):
                key_cache = kv_cache[0]
                block_size = key_cache.shape[1]

                # ALL positions in the cache
                all_positions = torch.arange(
                    0, total_seq_len,
                    device=key_cache.device,
                    dtype=torch.long,
                )

                logical_blocks = all_positions // block_size
                block_offsets = all_positions % block_size
                physical_blocks = logical_blocks  # identity for single-request

                # Read ALL keys (this is a copy via fancy indexing)
                all_keys = key_cache[
                    physical_blocks, block_offsets
                ]  # [total_seq_len, kv_heads, head_dim]

                # Step 1: Undo RoPE — strip position encoding
                raw_keys = apply_rope_rotation(
                    all_keys, all_positions, head_dim, rope_theta,
                    inverse=True,
                )

                # Step 2: Redo RoPE — reapply position encoding
                _ = apply_rope_rotation(
                    raw_keys, all_positions, head_dim, rope_theta,
                    inverse=False,
                )

                # Write ORIGINALS back — cache is unchanged
                key_cache[physical_blocks, block_offsets] = all_keys

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    deletion_latency_ms = (end_time - start_time) * 1000.0

    return {
        "deletion_latency_ms": deletion_latency_ms,
        "rope_storage": rope_storage,
        "delete_start": delete_start,
        "delete_end": delete_end,
        "total_tokens": total_seq_len,
    }


def kvlobotomy_full_delete(worker, delete_start, delete_end, rope_storage,
                            rope_theta, head_dim, num_kv_heads, total_seq_len):
    """Full delete: remove segment B and compact C+D into place.

    Unlike kvlobotomy_delete() which only does RoPE correction, this function
    actually moves C+D's keys AND values to fill the gap left by B.

    For post-RoPE: delta-rotate C+D's keys, then copy keys+values.
    For pre-RoPE: copy keys+values only (no rotation needed).

    Args:
        worker: vLLM worker object (provides access to kv_caches)
        delete_start: start position of segment to delete (inclusive)
        delete_end: end position of segment to delete (exclusive)
        rope_storage: "pre" or "post"
        rope_theta: RoPE base frequency
        head_dim: attention head dimension
        num_kv_heads: number of KV heads
        total_seq_len: total sequence length before deletion

    Returns:
        dict with deletion_latency_ms, rotation_ms, copy_ms, and metadata
    """
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    delete_len = delete_end - delete_start
    tokens_after = total_seq_len - delete_end  # C+D tokens to move

    rotation_ms = 0.0
    copy_ms = 0.0

    if tokens_after > 0:
        kv_caches = worker.model_runner.kv_caches

        for layer_idx, kv_cache in enumerate(kv_caches):
            key_cache = kv_cache[0]   # [num_blocks, block_size, kv_heads, head_dim]
            val_cache = kv_cache[1]   # [num_blocks, block_size, kv_heads, head_dim]
            block_size = key_cache.shape[1]

            # Source positions (where C+D currently are)
            old_positions = torch.arange(
                delete_end, total_seq_len,
                device=key_cache.device, dtype=torch.long,
            )
            # Destination positions (where C+D should go after compaction)
            new_positions = old_positions - delete_len

            # Map to block addresses
            old_blocks = old_positions // block_size
            old_offsets = old_positions % block_size
            new_blocks = new_positions // block_size
            new_offsets = new_positions % block_size

            # For single-request experiment: block table is identity
            old_phys = old_blocks
            new_phys = new_blocks

            # Gather C+D's keys and values from old locations
            torch.cuda.synchronize()
            t_copy_start = time.perf_counter()

            c_keys = key_cache[old_phys, old_offsets]  # [N, kv_heads, head_dim]
            c_vals = val_cache[old_phys, old_offsets]  # [N, kv_heads, head_dim]

            # Post-RoPE: delta-rotate C+D's keys before placing them
            if rope_storage == "post":
                torch.cuda.synchronize()
                t_rot_start = time.perf_counter()

                c_keys = apply_delta_rotation(
                    c_keys, old_positions, -delete_len, head_dim, rope_theta,
                )

                torch.cuda.synchronize()
                t_rot_end = time.perf_counter()
                rotation_ms += (t_rot_end - t_rot_start) * 1000.0

            # Scatter C+D's data to new (compacted) locations
            key_cache[new_phys, new_offsets] = c_keys
            val_cache[new_phys, new_offsets] = c_vals

            torch.cuda.synchronize()
            t_copy_end = time.perf_counter()
            copy_ms += (t_copy_end - t_copy_start) * 1000.0

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    deletion_latency_ms = (end_time - start_time) * 1000.0

    # copy_ms includes rotation_ms for post-RoPE; separate them
    copy_only_ms = copy_ms - rotation_ms

    return {
        "deletion_latency_ms": deletion_latency_ms,
        "rotation_ms": rotation_ms,
        "copy_ms": copy_only_ms,
        "total_tokens_moved": tokens_after,
        "total_tokens_deleted": delete_len,
        "rope_storage": rope_storage,
        "delete_start": delete_start,
        "delete_end": delete_end,
    }
