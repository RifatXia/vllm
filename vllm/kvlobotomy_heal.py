# SPDX-License-Identifier: Apache-2.0
"""Hash table healing after KV cache deletion.

After kvlobotomy_full_delete() compacts the KV cache data on the GPU,
the prefix cache hash table on the scheduler side has stale entries.
This module selectively heals the hash table so that both A and C
get cache hits on subsequent queries.

Healing strategy:
  1. Compute block hashes for the OLD (ABC) and NEW (AC) token sequences
  2. Find the first block where hashes diverge (this is where B starts,
     or the block straddling the A/B boundary)
  3. From that block onward: remove old hash entries, collect physical blocks
  4. Re-register those physical blocks with the new (AC) hashes
  5. Free any tail blocks that are beyond the new sequence length

Usage from experiment script:
    from vllm.kvlobotomy_heal import heal_prefix_cache_after_delete
    result = heal_prefix_cache_after_delete(llm, abc_token_ids, delete_start, delete_end)
"""

import logging
from collections.abc import Callable, Sequence

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    hash_block_tokens,
    make_block_hash_with_group_id,
)

logger = logging.getLogger(__name__)


def _compute_block_hashes(
    token_ids: Sequence[int],
    block_size: int,
    hash_fn: Callable,
) -> list[BlockHash]:
    """Compute chained block hashes for a token sequence.

    Only full blocks (exactly block_size tokens) are hashed.
    Each block's hash depends on the previous block's hash (chain).

    Args:
        token_ids: Full token sequence.
        block_size: Number of tokens per block.
        hash_fn: The hash function (e.g. sha256_cbor).

    Returns:
        List of BlockHash, one per full block.
    """
    hashes: list[BlockHash] = []
    prev_hash: BlockHash | None = None
    num_full_blocks = len(token_ids) // block_size
    for i in range(num_full_blocks):
        start = i * block_size
        block_tokens = tuple(token_ids[start:start + block_size])
        block_hash = hash_block_tokens(hash_fn, prev_hash, block_tokens, None)
        hashes.append(block_hash)
        prev_hash = block_hash
    return hashes


def _get_engine_core(llm):
    """Navigate from LLM object to the EngineCore instance.

    In offline (LLM) mode, the engine core client is an InprocClient
    that wraps the actual EngineCore in-process.
    """
    engine_core_client = llm.llm_engine.engine_core
    if hasattr(engine_core_client, "engine_core"):
        return engine_core_client.engine_core
    return engine_core_client


def get_physical_block_table(
    llm,
    abc_token_ids: list[int],
) -> list[int]:
    """Get the physical block IDs for the ABC sequence from the hash table.

    Must be called BEFORE kvlobotomy_full_delete, while the hash table
    still has the original ABC entries. The returned list maps logical
    block index to physical block ID:

        block_table[logical_idx] = physical_block_id

    This is needed because vLLM's paged KV cache allocates physical blocks
    from a free list -- physical block 0 is the null block, and actual data
    blocks could be [3, 4, 5, ...] depending on allocation order.

    Handles partial blocks: the hash table only tracks full blocks (block_size
    tokens), but the sequence may have a partial block at the end. This
    function recovers the partial block's ID by scanning the free list tail
    (recently freed blocks) for the one without a hash entry.

    Args:
        llm: vLLM LLM instance.
        abc_token_ids: Token IDs of the ABC populate sequence.

    Returns:
        List of physical block IDs, one per block (full + partial) in
        the sequence.
    """
    core = _get_engine_core(llm)
    block_pool = core.scheduler.kv_cache_manager.block_pool
    block_size = block_pool.hash_block_size
    hash_fn = core.caching_hash_fn

    if hash_fn is None:
        raise RuntimeError("Prefix caching hash function not available.")

    hashes = _compute_block_hashes(abc_token_ids, block_size, hash_fn)

    # Use the first KV cache group (standard attention = group 0)
    coordinator = core.scheduler.kv_cache_manager.coordinator
    group_id = coordinator.single_type_managers[0].kv_cache_group_id

    # Phase 1: Get full block IDs from hash table
    block_table: list[int] = []
    for i, block_hash in enumerate(hashes):
        hash_with_group = make_block_hash_with_group_id(block_hash, group_id)
        block = block_pool.cached_block_hash_to_block.get_one_block(
            hash_with_group
        )
        if block is not None:
            block_table.append(block.block_id)
        else:
            logger.warning(
                "Block %d not found in hash table during block_table lookup", i
            )
            block_table.append(i)  # fallback to identity (best effort)

    # Phase 2: Handle partial block at end of sequence.
    # The hash table only tracks full blocks. If the sequence doesn't end
    # on a block boundary, there's one more block allocated for the
    # partial tail that has no hash entry.
    total_blocks_needed = (len(abc_token_ids) + block_size - 1) // block_size
    if total_blocks_needed > len(block_table):
        known_ids = set(block_table)
        partial_block_id = _find_partial_block(block_pool, known_ids)
        if partial_block_id is not None:
            block_table.append(partial_block_id)
            logger.info("Partial block found: physical_id=%d", partial_block_id)
        else:
            # Fallback: assume it follows the last full block sequentially
            fallback_id = block_table[-1] + 1 if block_table else 0
            block_table.append(fallback_id)
            logger.warning(
                "Could not find partial block, using fallback id=%d",
                fallback_id,
            )

    logger.info(
        "Physical block table: %d blocks (need %d), IDs: %s",
        len(block_table),
        total_blocks_needed,
        block_table[:10],
    )

    return block_table


def _find_partial_block(block_pool, known_full_ids: set[int]) -> int | None:
    """Find the partial block among recently freed blocks.

    After the populate request completes, all its blocks (full + partial)
    are freed to the END of the free list via append_n(). The full blocks
    have block_hash set; the partial block has block_hash=None.

    We scan backwards from the free list tail to find the one block that:
    - Is NOT in our known full-block set
    - Has no block_hash (partial blocks are never cached)

    Args:
        block_pool: The BlockPool instance.
        known_full_ids: Set of physical block IDs for full blocks.

    Returns:
        Physical block ID of the partial block, or None if not found.
    """
    tail_sentinel = block_pool.free_block_queue.fake_free_list_tail
    curr = tail_sentinel.prev_free_block

    # Scan backwards through the recently freed blocks.
    # The populate's blocks are at the end. We scan at most
    # len(known_full_ids)+10 entries to find the partial block nearby.
    scan_limit = len(known_full_ids) + 10
    for _ in range(scan_limit):
        if curr is None or curr.block_id == -1:  # reached fake head
            break
        if curr.block_id not in known_full_ids and curr.block_hash is None:
            return curr.block_id
        curr = curr.prev_free_block

    return None


def heal_prefix_cache_after_delete(
    llm,
    abc_token_ids: list[int],
    delete_start: int,
    delete_end: int,
) -> dict:
    """Heal prefix cache hash table after segment deletion.

    After kvlobotomy_full_delete() has compacted the KV cache (removed B,
    shifted C into B's gap, zeroed tail), this function updates the hash
    table so that subsequent queries get cache hits on both A and C.

    The function:
      - Keeps A's hash entries intact (data unchanged in cache)
      - Removes hash entries for blocks that changed (B's blocks, C's old blocks)
      - Re-registers C's blocks at their new positions with correct hashes
      - Leaves tail blocks (zeroed) without hash entries (effectively freed)

    Args:
        llm: vLLM LLM instance (offline mode, in-process engine core).
        abc_token_ids: Token IDs of the original ABC populate sequence.
        delete_start: First token index of the deleted segment (inclusive).
        delete_end: First token index after the deleted segment (exclusive).

    Returns:
        Dict with healing metadata (blocks kept, removed, registered, freed).
    """
    core = _get_engine_core(llm)
    block_pool = core.scheduler.kv_cache_manager.block_pool
    block_size = block_pool.hash_block_size
    hash_fn = core.caching_hash_fn

    if hash_fn is None:
        raise RuntimeError(
            "Prefix caching hash function not available. "
            "Is enable_prefix_caching=True?"
        )

    # Construct AC token IDs (what the cache physically contains after deletion)
    ac_token_ids = list(abc_token_ids[:delete_start]) + list(
        abc_token_ids[delete_end:]
    )

    # Compute block hashes for old (ABC) and new (AC) sequences
    old_hashes = _compute_block_hashes(abc_token_ids, block_size, hash_fn)
    new_hashes = _compute_block_hashes(ac_token_ids, block_size, hash_fn)

    # Find first block where hashes diverge.
    # All blocks before this index are identical (pure A content, unchanged).
    first_changed = min(len(old_hashes), len(new_hashes))
    for i in range(first_changed):
        if old_hashes[i] != new_hashes[i]:
            first_changed = i
            break

    # Get KV cache group IDs (one per attention type; typically just [0])
    coordinator = core.scheduler.kv_cache_manager.coordinator
    group_ids = [
        mgr.kv_cache_group_id for mgr in coordinator.single_type_managers
    ]

    blocks_removed = 0
    blocks_registered = 0
    blocks_freed = 0

    for group_id in group_ids:
        # Phase 1: Remove old hash entries from first_changed onward,
        #          collecting the physical KVCacheBlock objects in order.
        old_blocks_ordered = []
        for i in range(first_changed, len(old_hashes)):
            hash_with_group = make_block_hash_with_group_id(
                old_hashes[i], group_id
            )
            block = block_pool.cached_block_hash_to_block.get_one_block(
                hash_with_group
            )
            if block is not None:
                block_pool.cached_block_hash_to_block.pop(
                    hash_with_group, block.block_id
                )
                block.reset_hash()
                old_blocks_ordered.append(block)
                blocks_removed += 1
            else:
                old_blocks_ordered.append(None)
                logger.warning(
                    "Block %d (group %d) not found in hash table during heal",
                    i, group_id,
                )

        # Phase 2: Re-register blocks with new (AC) hashes.
        # Block at position first_changed in the old sequence is the same
        # physical block that now holds data for position first_changed
        # in the new sequence (kvlobotomy_full_delete wrote to the same
        # physical locations).
        num_to_register = len(new_hashes) - first_changed
        for j in range(num_to_register):
            if j >= len(old_blocks_ordered):
                break
            block = old_blocks_ordered[j]
            if block is not None:
                hash_with_group = make_block_hash_with_group_id(
                    new_hashes[first_changed + j], group_id
                )
                block.block_hash = hash_with_group
                block_pool.cached_block_hash_to_block.insert(
                    hash_with_group, block
                )
                blocks_registered += 1

        # Phase 3: Tail blocks (beyond new sequence) are zeroed out.
        # Their hash entries were already removed in phase 1; they stay
        # in the free list without hash registrations.
        for j in range(num_to_register, len(old_blocks_ordered)):
            if old_blocks_ordered[j] is not None:
                blocks_freed += 1

    result = {
        "blocks_kept": first_changed * len(group_ids),
        "blocks_removed": blocks_removed,
        "blocks_registered": blocks_registered,
        "blocks_freed": blocks_freed,
        "old_total_blocks": len(old_hashes),
        "new_total_blocks": len(new_hashes),
        "first_changed_block": first_changed,
    }

    logger.info(
        "Hash table healed: kept=%d, removed=%d, registered=%d, freed=%d "
        "(first_changed=%d, old=%d blocks, new=%d blocks)",
        result["blocks_kept"],
        blocks_removed,
        blocks_registered,
        blocks_freed,
        first_changed,
        len(old_hashes),
        len(new_hashes),
    )

    return result
