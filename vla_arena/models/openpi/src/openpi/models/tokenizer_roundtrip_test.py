"""Diagnostic script: FAST tokenizer encode → decode round-trip.

Tests whether extract_actions can faithfully recover actions that were
encoded by tokenize, under both full-sequence and inference-time conditions.
"""

import numpy as np
from openpi.models import tokenizer as _tokenizer


def test_full_sequence_roundtrip(
    action_horizon: int = 10, action_dim: int = 7, seed: int = 42
):
    """Round-trip using the full token sequence from tokenize() — includes
    both the prefix (Task/State) and postfix (Action tokens)."""
    rng = np.random.RandomState(seed)
    prompt = "pick up the red block"
    state = rng.randn(action_dim).astype(np.float32) * 0.5
    actions = rng.randn(action_horizon, action_dim).astype(np.float32) * 0.3

    tok = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_mask, ar_mask, loss_mask = tok.tokenize(prompt, state, actions)

    print("=== Test 1: Full-sequence round-trip ===")
    print(f"Non-zero tokens: {np.count_nonzero(tokens)}")

    recovered = tok.extract_actions(tokens, action_horizon, action_dim)
    mse = np.mean((recovered - actions) ** 2)
    max_err = np.max(np.abs(recovered - actions))
    print(f"Shape: {recovered.shape}  MSE: {mse:.6f}  MaxErr: {max_err:.6f}")
    print(f"  Original : {actions[0, :4]}")
    print(f"  Recovered: {recovered[0, :4]}")
    print()
    return mse


def test_inference_time_roundtrip(
    action_horizon: int = 10, action_dim: int = 7, seed: int = 42
):
    """Simulate inference-time: the model only generates the postfix tokens
    (Action: <FAST_tokens> | EOS), padded with zeros to max_decoding_steps."""
    rng = np.random.RandomState(seed)
    prompt = "pick up the red block"
    state = rng.randn(action_dim).astype(np.float32) * 0.5
    actions = rng.randn(action_horizon, action_dim).astype(np.float32) * 0.3

    tok = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_mask, ar_mask, loss_mask = tok.tokenize(prompt, state, actions)

    # Identify where the postfix starts (ar_mask transitions from 0 to 1)
    ar_np = np.array(ar_mask)
    postfix_start = int(np.argmax(ar_np == 1))

    # Build inference-time token array: only postfix + zero padding
    max_decoding_steps = 256
    inference_tokens = np.zeros(max_decoding_steps, dtype=tokens.dtype)
    postfix = tokens[postfix_start:]
    postfix_nonzero = postfix[postfix != 0]
    inference_tokens[: len(postfix_nonzero)] = postfix_nonzero

    print("=== Test 2: Inference-time round-trip (postfix only) ===")
    print(f"Postfix starts at index : {postfix_start}")
    print(f"Postfix non-zero tokens : {len(postfix_nonzero)}")
    print(f"Inference token array   : {inference_tokens[:len(postfix_nonzero)+3]}")

    # Extract FAST token IDs via the tokenizer's private method
    fast_ids = tok._extract_fast_token_ids(inference_tokens)
    pg_vocab = tok._paligemma_tokenizer.vocab_size()
    fast_lower = pg_vocab - 1 - tok._fast_skip_tokens - tok._fast_tokenizer.vocab_size + 1
    n_text = np.sum((inference_tokens > 0) & (inference_tokens < fast_lower))
    print(f"FAST tokens in range    : {len(fast_ids)}")
    print(f"Text tokens (low range) : {n_text}")

    recovered = tok.extract_actions(inference_tokens, action_horizon, action_dim)
    mse = np.mean((recovered - actions) ** 2)
    max_err = np.max(np.abs(recovered - actions))
    print(f"Shape: {recovered.shape}  MSE: {mse:.6f}  MaxErr: {max_err:.6f}")
    print(f"  Original : {actions[0, :4]}")
    print(f"  Recovered: {recovered[0, :4]}")
    print()
    return mse


def test_perturbed_bpe_length(
    action_horizon: int = 10, action_dim: int = 7, seed: int = 42
):
    """Verify graceful degradation when BPE decode yields wrong-length output.
    We simulate this by dropping 1-2 FAST tokens from the sequence."""
    rng = np.random.RandomState(seed)
    prompt = "pick up the red block"
    state = rng.randn(action_dim).astype(np.float32) * 0.5
    actions = rng.randn(action_horizon, action_dim).astype(np.float32) * 0.3

    tok = _tokenizer.FASTTokenizer(max_len=256)
    tokens, _, _, _ = tok.tokenize(prompt, state, actions)

    original_fast_ids = tok._extract_fast_token_ids(tokens)

    # We need *positional* indices to zero-out a token, so recompute the range
    # bounds here (extract_fast_token_ids returns values, not positions).
    pg_vocab = tok._paligemma_tokenizer.vocab_size()
    fast_upper = pg_vocab - 1 - tok._fast_skip_tokens
    fast_lower = fast_upper - tok._fast_tokenizer.vocab_size + 1
    fast_indices = np.where(
        (tokens >= fast_lower) & (tokens <= fast_upper)
    )[0]

    print("=== Test 3: Perturbed BPE length (drop last FAST token) ===")
    print(f"Original FAST token count: {len(original_fast_ids)}")

    perturbed = tokens.copy()
    perturbed[fast_indices[-1]] = 0

    n_fast_after = len(tok._extract_fast_token_ids(perturbed))
    print(f"Perturbed FAST token count: {n_fast_after}")

    recovered = tok.extract_actions(perturbed, action_horizon, action_dim)
    mse = np.mean((recovered - actions) ** 2)
    max_err = np.max(np.abs(recovered - actions))
    is_zero = np.allclose(recovered, 0)
    print(f"Shape: {recovered.shape}  MSE: {mse:.6f}  MaxErr: {max_err:.6f}  AllZero: {is_zero}")
    print(f"  Original : {actions[0, :4]}")
    print(f"  Recovered: {recovered[0, :4]}")
    print()
    return mse


if __name__ == "__main__":
    mse1 = test_full_sequence_roundtrip()
    mse2 = test_inference_time_roundtrip()
    mse3 = test_perturbed_bpe_length()

    print("=== Summary ===")
    print(f"Full-sequence MSE   : {mse1:.6f}")
    print(f"Inference-time MSE  : {mse2:.6f}")
    print(f"Perturbed-token MSE : {mse3:.6f}")
