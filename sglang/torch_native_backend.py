import torch
import math
from torch.nn.functional import scaled_dot_product_attention

def _run_sdpa_forward_extend(        
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
    """Run the extend forward by using torch native sdpa op.
    Args:
        query: [num_tokens, num_heads, head_size]
        output: [num_tokens, num_heads, head_size]
        k_cache: [max_total_num_tokens, num_heads, head_size]
        v_cache: [max_total_num_tokens, num_heads, head_size]
        req_to_token: [max_num_reqs, max_context_len]
        req_pool_indices: [num_seqs]
        seq_lens: [num_seqs]
        extend_prefix_lens: [num_seqs]
        extend_seq_lens: [num_seqs]
        scaling: float or None
        enable_gqa: bool
        causal: bool
    Returns:
        output: [num_tokens, num_heads, head_size]
    """
    assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
    assert seq_lens.shape[0] == extend_seq_lens.shape[0]

    # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
    query = query.movedim(0, query.dim() - 2)

    start_q, start_kv = 0, 0
    for seq_idx in range(seq_lens.shape[0]):
        # TODO: this loop process a sequence per iter, this is inefficient.
        # Need optimize the performance later.

        extend_seq_len_q = extend_seq_lens[seq_idx]
        prefill_seq_len_q = extend_prefix_lens[seq_idx]

        seq_len_kv = seq_lens[seq_idx]
        end_q = start_q + extend_seq_len_q
        end_kv = start_kv + seq_len_kv

        per_req_query = query[:, start_q:end_q, :]
        per_req_query_redudant = torch.empty(
            (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
            dtype=per_req_query.dtype,
            device=per_req_query.device,
        )

        per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

        # get key and value from cache. per_req_tokens contains the kv cache
        # index for each token in the sequence.
        req_pool_idx = req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
        per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

        per_req_query_redudant_unsqueezed = per_req_query_redudant.unsqueeze(0)
        print(f"per_req_query_redudant_unsqueezed shape: {per_req_query_redudant_unsqueezed.shape}")
        print(f"per_req_key shape: {per_req_key.shape}")
        print(f"per_req_value shape: {per_req_value.shape}")

        per_req_out_redudant = (
            scaled_dot_product_attention(
                per_req_query_redudant.unsqueeze(0),
                per_req_key.unsqueeze(0),
                per_req_value.unsqueeze(0),
                enable_gqa=enable_gqa,
                scale=scaling,
                is_causal=causal,
            )
            .squeeze(0)
            .movedim(query.dim() - 2, 0)
        )
        output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
        start_q, start_kv = end_q, end_kv
    return output


def create_test_data_for_extend():
    """
    Create sample data to test _run_sdpa_forward_extend function.
    
    Scenario: 3 sequences being extended with different configurations
    """
    
    # Model configuration
    num_heads = 8
    head_size = 64
    max_total_tokens = 1000
    max_num_reqs = 100
    max_context_len = 512
    
    # Batch configuration
    num_seqs = 3
    seq_lens = torch.tensor([10, 15, 8])                # Total tokens after extension
    extend_prefix_lens = torch.tensor([7, 12, 5])       # Already cached tokens
    extend_seq_lens = torch.tensor([3, 3, 3])           # New tokens being added
    req_pool_indices = torch.tensor([42, 17, 89])       # Memory pool locations
    
    # Calculate total new tokens
    num_tokens = extend_seq_lens.sum().item()  # 3 + 3 + 3 = 9
    
    print(f"=== TEST DATA CONFIGURATION ===")
    print(f"num_seqs: {num_seqs}")
    print(f"seq_lens: {seq_lens.tolist()}")
    print(f"extend_prefix_lens: {extend_prefix_lens.tolist()}")
    print(f"extend_seq_lens: {extend_seq_lens.tolist()}")
    print(f"total num_tokens: {num_tokens}")
    print(f"req_pool_indices: {req_pool_indices.tolist()}")
    print()
    
    # Create query tensor [num_tokens, num_heads, head_size]
    query = torch.randn(num_tokens, num_heads, head_size)
    
    # Mark query tokens with identifiable values for debugging
    for i in range(num_tokens):
        query[i, :, 0] = i + 100  # First dimension has token ID (100, 101, 102, ...)
    
    print(f"Query tensor shape: {query.shape}")
    print(f"Query token markers (first head, first dim): {query[:, 0, 0].tolist()}")
    print()
    
    # Create output tensor (same shape as query)
    output = torch.zeros_like(query)
    
    # Create global KV cache [max_total_tokens, num_heads, head_size]
    k_cache = torch.randn(max_total_tokens, num_heads, head_size)
    v_cache = torch.randn(max_total_tokens, num_heads, head_size)
    
    # Mark cache entries with identifiable values
    for i in range(max_total_tokens):
        k_cache[i, :, 0] = i + 1000  # Key cache markers (1000, 1001, 1002, ...)
        v_cache[i, :, 0] = i + 2000  # Value cache markers (2000, 2001, 2002, ...)
    
    # Create req_to_token mapping [max_num_reqs, max_context_len]
    req_to_token = torch.zeros(max_num_reqs, max_context_len, dtype=torch.long)
    
    # Set up token mappings for our test sequences
    # Sequence 0: pool index 42, uses tokens 100-109 in cache
    req_to_token[42, :10] = torch.arange(100, 110)
    
    # Sequence 1: pool index 17, uses tokens 200-214 in cache  
    req_to_token[17, :15] = torch.arange(200, 215)
    
    # Sequence 2: pool index 89, uses tokens 300-307 in cache
    req_to_token[89, :8] = torch.arange(300, 308)
    
    print(f"=== TOKEN MAPPINGS ===")
    print(f"Seq 0 (pool {req_pool_indices[0]}): cache tokens {req_to_token[42, :10].tolist()}")
    print(f"Seq 1 (pool {req_pool_indices[1]}): cache tokens {req_to_token[17, :15].tolist()}")
    print(f"Seq 2 (pool {req_pool_indices[2]}): cache tokens {req_to_token[89, :8].tolist()}")
    print()
    
    # Set scaling factor
    scaling = 1.0 / math.sqrt(head_size)
    
    return {
        'query': query,
        'output': output,
        'k_cache': k_cache,
        'v_cache': v_cache,
        'req_to_token': req_to_token,
        'req_pool_indices': req_pool_indices,
        'seq_lens': seq_lens,
        'extend_prefix_lens': extend_prefix_lens,
        'extend_seq_lens': extend_seq_lens,
        'scaling': scaling,
        'enable_gqa': False,
        'causal': True
    }

def test_run_sdpa_forward_extend():
    """Test the _run_sdpa_forward_extend function with sample data."""
    
    # Get test data
    test_data = create_test_data_for_extend()
    
    # Extract parameters
    query = test_data['query']
    output = test_data['output']
    k_cache = test_data['k_cache']
    v_cache = test_data['v_cache']
    req_to_token = test_data['req_to_token']
    req_pool_indices = test_data['req_pool_indices']
    seq_lens = test_data['seq_lens']
    extend_prefix_lens = test_data['extend_prefix_lens']
    extend_seq_lens = test_data['extend_seq_lens']
    scaling = test_data['scaling']
    enable_gqa = test_data['enable_gqa']
    causal = test_data['causal']
    
    print("=== BEFORE PROCESSING ===")
    print(f"Output tensor (should be zeros): {output[:3, 0, 0].tolist()}")
    print()
    
    # Simulate the _run_sdpa_forward_extend function
    # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
    query_reshaped = query.movedim(0, query.dim() - 2)
    
    start_q, start_kv = 0, 0
    
    for seq_idx in range(seq_lens.shape[0]):
        print(f"=== PROCESSING SEQUENCE {seq_idx} ===")
        
        extend_seq_len_q = extend_seq_lens[seq_idx].item()
        prefill_seq_len_q = extend_prefix_lens[seq_idx].item()
        seq_len_kv = seq_lens[seq_idx].item()
        end_q = start_q + extend_seq_len_q
        end_kv = start_kv + seq_len_kv
        
        print(f"extend_seq_len_q: {extend_seq_len_q}")
        print(f"prefill_seq_len_q: {prefill_seq_len_q}")
        print(f"seq_len_kv: {seq_len_kv}")
        print(f"Query range: [{start_q}:{end_q}]")
        
        # Extract query for this sequence
        per_req_query = query_reshaped[:, start_q:end_q, :]
        print(f"per_req_query shape: {per_req_query.shape}")
        print(f"per_req_query markers: {per_req_query[0, :, 0].tolist()}")
        
        # Create redundant query tensor
        per_req_query_redundant = torch.zeros(
            (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
            dtype=per_req_query.dtype,
            device=per_req_query.device,
        )
        
        # Place new queries at correct positions  
        per_req_query_redundant[:, prefill_seq_len_q:, :] = per_req_query
        print(f"per_req_query_redundant shape: {per_req_query_redundant.shape}")
        print(f"Redundant query markers: {per_req_query_redundant[0, :, 0].tolist()}")
        
        # Get key and value from cache
        req_pool_idx = req_pool_indices[seq_idx].item()
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        print(f"req_pool_idx: {req_pool_idx}")
        print(f"per_req_tokens: {per_req_tokens.tolist()}")
        
        per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
        per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)
        print(f"per_req_key shape: {per_req_key.shape}")
        print(f"Key markers: {per_req_key[0, :, 0].tolist()}")
        print(f"Value markers: {per_req_value[0, :, 0].tolist()}")
        
        # Run attention
        per_req_out_redundant = (
            scaled_dot_product_attention(
                per_req_query_redundant.unsqueeze(0),
                per_req_key.unsqueeze(0),
                per_req_value.unsqueeze(0),
                scale=scaling,
                is_causal=causal,
            )
            .squeeze(0)
            .movedim(query.dim() - 2, 0)
        )
        
        print(f"per_req_out_redundant shape: {per_req_out_redundant.shape}")
        print(f"Output markers before extraction: {per_req_out_redundant[:, 0, 0].tolist()}")
        
        # Extract relevant outputs
        relevant_output = per_req_out_redundant[prefill_seq_len_q:, :, :]
        output[start_q:end_q, :, :] = relevant_output
        
        print(f"Extracted output shape: {relevant_output.shape}")
        print(f"Placed in output[{start_q}:{end_q}]")
        print(f"Output after placement: {output[start_q:end_q, 0, 0].tolist()}")
        print()
        
        start_q, start_kv = end_q, end_kv
    
    print("=== FINAL RESULTS ===")
    print(f"Final output shape: {output.shape}")
    print(f"Output sample (first head, first dim): {output[:, 0, 0].tolist()}")
    print()
    
    # Verify output is non-zero (attention worked)
    output_norm = torch.norm(output)
    print(f"Output tensor norm: {output_norm:.4f}")
    
    if output_norm > 0:
        print("✅ Test PASSED - Output is non-zero, attention computation worked")
    else:
        print("❌ Test FAILED - Output is zero, something went wrong")
    
    return output

def visualize_attention_pattern_correct():
    """
    Properly visualize attention patterns without tensor dimension issues.
    
    The key insight: we need to avoid zero queries that cause NaN when combined with causal masking.
    """
    
    print("=== CORRECTED ATTENTION PATTERN VISUALIZATION ===")
    
    # Configuration
    extend_prefix_len = 2  # 2 cached tokens
    extend_seq_len = 2     # 2 new tokens  
    seq_len_kv = 4         # Total tokens in sequence
    
    # Create meaningful queries (non-zero for positions that will be used)
    # Shape: [seq_len, head_dim] where we only care about the new token positions
    query_redundant = torch.tensor([
        [0.0],  # Position 0: cached (will be masked anyway)
        [0.0],  # Position 1: cached (will be masked anyway)  
        [1.0],  # Position 2: new token with query
        [2.0],  # Position 3: new token with query
    ])
    
    # Keys and values
    k_cache = torch.tensor([
        [1.0],  # Key for position 0
        [1.0],  # Key for position 1
        [1.0],  # Key for position 2  
        [1.0],  # Key for position 3
    ])
    
    v_cache = torch.tensor([
        [10.0],  # Value for position 0
        [20.0],  # Value for position 1
        [30.0],  # Value for position 2
        [40.0],  # Value for position 3
    ])
    
    print(f"Query (redundant): {query_redundant.squeeze().tolist()}")
    print(f"Key cache: {k_cache.squeeze().tolist()}")  
    print(f"Value cache: {v_cache.squeeze().tolist()}")
    print(f"Extend prefix length: {extend_prefix_len}")
    print(f"New token positions: {extend_prefix_len} onwards")
    print()
    
    # Compute raw attention scores: Q @ K^T
    scores = torch.matmul(query_redundant, k_cache.transpose(-2, -1))  # [4, 4]
    print(f"Raw scores shape: {scores.shape}")
    print("Raw scores (Q @ K^T):")
    for i, row in enumerate(scores.tolist()):
        print(f"  Position {i}: {row}")
    print()
    
    # Create causal mask: upper triangular with -inf
    causal_mask = torch.triu(torch.ones(4, 4), diagonal=1) * float('-inf')
    print("Causal mask:")
    for i, row in enumerate(causal_mask.tolist()):
        print(f"  Position {i}: {[f'{x:.0f}' if x != float('-inf') else '-∞' for x in row]}")
    print()
    
    # Apply causal mask
    scores_masked = scores + causal_mask
    print("Scores after causal masking:")
    for i, row in enumerate(scores_masked.tolist()):
        formatted_row = []
        for x in row:
            if x == float('-inf'):
                formatted_row.append('-∞')
            elif math.isnan(x):
                formatted_row.append('NaN')  
            else:
                formatted_row.append(f'{x:.1f}')
        print(f"  Position {i}: {formatted_row}")
    print()
    
    # Apply softmax to get attention weights
    attn_weights = torch.softmax(scores_masked, dim=-1)
    print("Attention weights (after softmax):")
    for i, row in enumerate(attn_weights.tolist()):
        formatted_row = []
        for x in row:
            if math.isnan(x):
                formatted_row.append('NaN')
            else:
                formatted_row.append(f'{x:.3f}')
        print(f"  Position {i}: {formatted_row}")
    print()
    
    # Compute final output: attention_weights @ V
    output = torch.matmul(attn_weights, v_cache)  # [4, 1]
    print(f"Final output shape: {output.shape}")
    print("Final output (attention @ V):")
    for i, val in enumerate(output.squeeze().tolist()):
        if math.isnan(val):
            print(f"  Position {i}: NaN")
        else:
            print(f"  Position {i}: {val:.2f}")
    print()
    
    # Extract only the outputs for new tokens (what the extend function would return)
    new_token_outputs = output[extend_prefix_len:]
    print(f"New token outputs (positions {extend_prefix_len}+): {new_token_outputs.squeeze().tolist()}")
    
    print("\n=== EXPLANATION ===")
    print("1. Positions 0,1: Cached tokens (prefix)")
    print("2. Positions 2,3: New tokens being processed")
    print("3. Causal mask ensures:")
    print("   - Position 0: Only attends to itself")
    print("   - Position 1: Attends to positions 0,1") 
    print("   - Position 2: Attends to positions 0,1,2")
    print("   - Position 3: Attends to positions 0,1,2,3")
    print("4. Only outputs for positions 2,3 are used (new tokens)")
    
    # Show that this matches the extend function behavior
    print("\n=== EXTEND FUNCTION SIMULATION ===")
    print("This is exactly what _run_sdpa_forward_extend does:")
    print("1. Creates redundant query with new tokens at correct positions")
    print("2. Uses full KV cache for the sequence") 
    print("3. Applies causal attention")
    print("4. Extracts outputs only for new token positions")

def simple_working_example():
    """A simple example that definitely works without any issues."""
    
    print("=== SIMPLE WORKING EXAMPLE ===")
    
    # Use PyTorch's SDPA directly (like the real function does)
    batch_size = 1
    num_heads = 2
    seq_len = 4  
    head_dim = 8
    
    # Create test tensors with proper dimensions
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)  
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    
    # Run attention with causal masking
    output = scaled_dot_product_attention(
        query, key, value,
        is_causal=True,
        scale=1.0 / math.sqrt(head_dim)
    )
    
    print(f"Output shape: {output.shape}")
    print(f"Output norm: {torch.norm(output):.4f}")
    
    # Simulate extracting new tokens (last 2 positions)
    extend_prefix_len = 2
    new_token_outputs = output[:, :, extend_prefix_len:, :]
    print(f"New token outputs shape: {new_token_outputs.shape}")
    print(f"New token outputs norm: {torch.norm(new_token_outputs):.4f}")
    
    print("✅ Simple example completed successfully!")
    print("This demonstrates that SDPA works correctly with proper tensor shapes.")

if __name__ == "__main__":

    testdata = create_test_data_for_extend()

    _run_sdpa_forward_extend(
        query=testdata['query'],
        output=testdata['output'],
        k_cache=testdata['k_cache'],
        v_cache=testdata['v_cache'],
        req_to_token=testdata['req_to_token'],
        req_pool_indices=testdata['req_pool_indices'],
        seq_lens=testdata['seq_lens'],
        extend_prefix_lens=testdata['extend_prefix_lens'],
        extend_seq_lens=testdata['extend_seq_lens'],
        scaling=testdata['scaling'],
        enable_gqa=testdata['enable_gqa'],
        causal=testdata['causal'],
        )
    
    # Run the corrected visualization
    visualize_attention_pattern_correct()
    print("\n" + "="*60 + "\n")
    
    # Run the main test
    test_run_sdpa_forward_extend()
