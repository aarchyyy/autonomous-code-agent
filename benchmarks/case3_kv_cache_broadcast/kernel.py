# Patched GQA KV-Cache Repeat-Interleave Broadcast Kernel
def gqa_kv_broadcast_kernel(kv_tensor, num_query_heads=12, num_kv_heads=2):
    """
    GQA KV Cache repeat-interleave broadcast expansion.
    Expands kv_tensor of shape [batch, num_kv_heads, seq_len, head_dim]
    to [batch, num_query_heads, seq_len, head_dim] with independent head buffers.
    """
    if num_kv_heads <= 0 or num_query_heads <= 0:
        raise ValueError("Head counts must be positive integers")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    group_size = num_query_heads // num_kv_heads
    broadcasted = []

    for b_idx, batch_item in enumerate(kv_tensor):
        if len(batch_item) != num_kv_heads:
            raise ValueError(
                f"Batch {b_idx} has {len(batch_item)} KV heads, expected {num_kv_heads}"
            )
        batch_heads = []
        for kv_head in batch_item:
            for _ in range(group_size):
                # Copy inner [seq_len][head_dim] slices so downstream mutations do not alias across heads
                batch_heads.append([list(seq_vec) for seq_vec in kv_head])
        broadcasted.append(batch_heads)

    return broadcasted
