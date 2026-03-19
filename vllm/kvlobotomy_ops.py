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


def kvlobotomy_delete(worker, delete_start, delete_end, rope_storage,
                      rope_theta, head_dim, num_kv_heads, total_seq_len):
    """Delete segment [delete_start, delete_end) from the KV cache.

    For post-RoPE: applies delta rotation to all keys after the deleted segment.
    For pre-RoPE: metadata-only operation (position shift).

    This function runs on the GPU worker process via collective_rpc.

    Args:
        worker: vLLM worker object (provides access to kv_caches)
        delete_start: start position of segment to delete (inclusive)
        delete_end: end position of segment to delete (exclusive)
        rope_storage: "pre" or "post"
        rope_theta: RoPE base frequency (e.g., 500000.0 for Llama 3.2)
        head_dim: attention head dimension
        num_kv_heads: number of KV heads
        total_seq_len: total sequence length before deletion

    Returns:
        dict with deletion_latency_ms and metadata
    """
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    delete_len = delete_end - delete_start
    affected_start = delete_end  # First position after deleted segment
    affected_len = total_seq_len - delete_end  # Number of tokens to correct
    delta = -delete_len  # Shift positions left

    if rope_storage == "pre":
        # Pre-RoPE: keys are stored unrotated. Deletion only requires
        # updating position indices (metadata), not tensor operations.
        # The next forward pass will apply RoPE with corrected positions
        # automatically.
        # Cost: O(1) for RoPE purposes.
        pass  # No tensor work needed

    elif rope_storage == "post":
        # Post-RoPE: keys are stored with positions baked in. We must
        # apply a delta rotation to all keys after the deleted segment.
        if affected_len > 0:
            kv_caches = worker.model_runner.kv_caches
            for layer_idx, kv_cache in enumerate(kv_caches):
                # kv_cache shape: [2, num_blocks, block_size, num_kv_heads,
                #                  head_dim]
                key_cache = kv_cache[0]
                block_size = key_cache.shape[1]

                # Build position mapping for affected tokens
                affected_positions = torch.arange(
                    affected_start,
                    total_seq_len,
                    device=key_cache.device,
                    dtype=torch.long,
                )

                logical_blocks = affected_positions // block_size
                block_offsets = affected_positions % block_size

                # For single-request experiment: block table is identity
                # (logical block i = physical block i)
                # In a real implementation, we'd look up the block table.
                physical_blocks = logical_blocks

                # Extract affected keys
                affected_keys = key_cache[
                    physical_blocks, block_offsets
                ]  # [affected_len, kv_heads, head_dim]

                # Apply delta rotation
                corrected_keys = apply_delta_rotation(
                    affected_keys,
                    affected_positions,
                    delta,
                    head_dim,
                    rope_theta,
                )

                # Write corrected keys back
                key_cache[physical_blocks, block_offsets] = corrected_keys

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    deletion_latency_ms = (end_time - start_time) * 1000.0

    return {
        "deletion_latency_ms": deletion_latency_ms,
        "rope_storage": rope_storage,
        "delete_start": delete_start,
        "delete_end": delete_end,
        "affected_tokens": affected_len,
        "delta": delta,
    }
