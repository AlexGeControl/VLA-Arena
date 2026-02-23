# Copyright 2025 The VLA-Arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import jax
import numpy as np
import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download
import orbax.checkpoint as ocp
import sentencepiece
from scipy.fft import idct
from transformers import AutoProcessor


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download(
            'gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}
        )
        with path.open('rb') as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read()
            )

    def tokenize(
        self, prompt: str, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace('_', ' ').replace('\n', ' ')
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = (
                np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            )
            state_str = ' '.join(map(str, discretized_state))
            full_prompt = (
                f'Task: {cleaned_text}, State: {state_str};\nAction: '
            )
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(
                cleaned_text, add_bos=True
            ) + self._tokenizer.encode('\n')
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. '
                    'Consider increasing the `max_token_len` in your model config if this happens frequently.'
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    def __init__(
        self,
        max_len: int = 256,
        fast_tokenizer_path: str = 'physical-intelligence/fast',
    ):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download(
            'gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}
        )
        with path.open('rb') as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read()
            )

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_path, 
            trust_remote_code=True
        )

        # Skip last 128 tokens in PaliGemma vocab since they are special tokens
        self._fast_skip_tokens = 128  

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace('_', ' ')

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = (
            np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        )

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = ' '.join(map(str, discretized_state))
        prefix = f'Task: {cleaned_text}, State: {state_str};\n'
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(
                action_tokens
            )

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode('Action: ')
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode('|', add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(
            postfix_tokens
        )  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. '
                    'Consider increasing the `max_token_len` in your model config if this happens frequently.'
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )

    def extract_actions(
        self, tokens: np.ndarray, action_horizon: int, action_dim: int
    ) -> np.ndarray:
        """Extract continuous actions from raw PaliGemma token IDs.

        Combines two fixes over the upstream ``UniversalActionProcessor``:

        1. **Range-based token extraction** — see
           :meth:`_extract_fast_token_ids`.  We filter by numerical range
           instead of doing a SentencePiece decode→re-encode round-trip,
           which is lossy for high-range IDs.

        2. **Relaxed BPE decode** — see :meth:`_decode_dct_coefficients`.
           BPE decompression may yield slightly more or fewer characters than
           ``action_horizon * action_dim``.  We pad/truncate to the expected
           length instead of failing on reshape.
        """
        fast_ids = self._extract_fast_token_ids(tokens)

        if len(fast_ids) == 0:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        try:
            dct_coeff = self._decode_dct_coefficients(
                fast_ids, action_horizon, action_dim
            )
            return self._reconstruct_action_trajectory(dct_coeff)
        except Exception as e:
            logging.warning('FAST decode failed: %s', e)
            return np.zeros(
                (action_horizon, action_dim), dtype=np.float32
            )


    def _act_tokens_to_paligemma_tokens(
        self, tokens: np.ndarray | list[int]
    ) -> np.ndarray:
        """Map between FAST-native IDs and PaliGemma IDs (involution).

        Used by :meth:`tokenize` for the FAST→PaliGemma direction during
        training data preparation.  The inverse direction (PaliGemma→FAST)
        for inference is handled by :meth:`_extract_fast_token_ids`.
        """
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return (
            self._paligemma_tokenizer.vocab_size()
            - 1
            - self._fast_skip_tokens
            - tokens
        )


    def _extract_fast_token_ids(self, tokens: np.ndarray) -> np.ndarray:
        """Extract FAST action-token IDs from a raw PaliGemma token sequence.

        Pi-Zero-FAST encodes continuous robot actions as discrete tokens and
        injects them into PaliGemma's vocabulary.  The vocabulary layout is::

            PaliGemma vocab (size V)
            ┌──────────────────────────────────────────────────────┐
            │  text tokens  │  FAST action tokens  │  128 special  │
            │  0 .. L-1     │  L .. L+F-1          │  L+F .. V-1   │
            └──────────────────────────────────────────────────────┘
                            ↑                      ↑
                        fast_lower             fast_upper

        where ``V`` is the PaliGemma vocabulary size, ``F`` is the FAST BPE
        vocabulary size, and the last 128 positions are reserved PaliGemma
        special tokens.

        Given a mixed sequence of text *and* action PaliGemma IDs (as
        produced by the autoregressive LLM head), this method:

        1. Identifies which tokens fall inside the FAST action range.
        2. Converts the matching PaliGemma IDs into FAST-native IDs via
           the involution ``fast_id = fast_upper - pg_id``.

        The mapping is its own inverse (an involution)::

            pg_id  →  fast_id = (V - 1 - skip) - pg_id
            fast_id →  pg_id  = (V - 1 - skip) - fast_id

        Returns
        -------
        np.ndarray
            FAST-native token IDs (may be empty if no action tokens found).
        """
        pg_vocab = self._paligemma_tokenizer.vocab_size()
        fast_upper = pg_vocab - 1 - self._fast_skip_tokens
        fast_lower = fast_upper - self._fast_tokenizer.vocab_size + 1

        # Boolean mask: True for tokens inside the FAST action range
        mask = (tokens >= fast_lower) & (tokens <= fast_upper)
        pg_action_ids = tokens[mask]

        if len(pg_action_ids) == 0:
            return np.array([], dtype=np.int64)

        # Involution: convert PaliGemma IDs → FAST-native IDs
        return fast_upper - pg_action_ids


    def _decode_dct_coefficients(
        self,
        fast_ids: np.ndarray,
        action_horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        """Decode FAST token IDs into a matrix of quantized DCT coefficients.

        This reverses the FAST *encoding* pipeline, which works as follows
        during training::

            continuous actions            (H, D)  float
                 │  DCT along time axis
                 ▼
            DCT coefficients              (H, D)  float
                 │  × scale,  round
                 ▼
            quantized coefficients        (H, D)  int
                 │  − min_token  (shift to non-negative)
                 ▼
            Unicode codepoints            (H*D,)  int  ≥ 0
                 │  chr()  → character string
                 ▼
            character string              len H*D
                 │  BPE compression
                 ▼
            BPE token IDs                 (T,)    int,  T ≤ H*D

        This method performs the *reverse* path (bottom → up to "quantized
        coefficients"), i.e.::

            BPE token IDs  →  BPE decompress  →  character string
                           →  ord() per char  →  codepoints
                           →  + min_token     →  quantized DCT coefficients

        **Why BPE output length can differ from H*D:** BPE is a variable-rate
        compression scheme.  During *encoding*, H*D characters are compressed
        into T ≤ H*D token IDs.  During *decoding*, T token IDs decompress
        back to characters — but the autoregressive model may generate
        slightly more or fewer tokens than the training-time T, causing the
        decompressed string to be shorter or longer than H*D.  We handle this
        with explicit pad/truncate (the "relaxed decode" fix).

        Parameters
        ----------
        fast_ids : np.ndarray
            FAST-native BPE token IDs (as returned by
            :meth:`_extract_fast_token_ids`).
        action_horizon : int
            Number of time steps in the action trajectory (H).
        action_dim : int
            Dimensionality of each action vector (D).

        Returns
        -------
        np.ndarray, shape ``(action_horizon, action_dim)``
            Quantized DCT coefficients (still need ``/ scale`` and IDCT to
            recover continuous actions).
        """
        # BPE decompress: token IDs → character string
        bpe_decoded: str = self._fast_tokenizer.bpe_tokenizer.decode(
            fast_ids.tolist()
        )

        # Each character's Unicode codepoint encodes one quantized DCT
        # coefficient, shifted by −min_token during encoding to ensure
        # non-negative codepoints.  We reverse the shift here.
        dct_coeff = (
            np.array(list(map(ord, bpe_decoded)), dtype=np.float32)
            + self._fast_tokenizer.min_token
        )

        # Relaxed decode: pad or truncate to the expected flat length (H * D).
        expected_len = action_horizon * action_dim
        diff = expected_len - dct_coeff.shape[0]
        # Truncate if too long
        if diff < 0:
            dct_coeff = dct_coeff[:expected_len]
        # Pad if too short
        elif diff > 0:
            dct_coeff = np.pad(
                dct_coeff, (0, diff), mode='constant', constant_values=0
            )

        return dct_coeff.reshape(action_horizon, action_dim)


    def _reconstruct_action_trajectory(
        self, dct_coeff: np.ndarray
    ) -> np.ndarray:
        """Recover a continuous action trajectory from quantized DCT coefficients.

        This is the final stage of FAST decoding.  The quantized DCT
        coefficient matrix produced by :meth:`_decode_dct_coefficients` still
        carries two artifacts of the encoding process:

        1. **Scaling** — during encoding, the DCT output was multiplied by
           ``scale`` (default 10) before rounding to integers, which
           preserves one extra decimal digit of precision.  We undo this by
           dividing by ``scale``.
        2. **DCT basis** — the coefficients live in the frequency domain of
           the Discrete Cosine Transform applied along the *time* axis.
           We apply the inverse DCT (IDCT) to recover the original
           time-domain action trajectory.

        Parameters
        ----------
        dct_coeff : np.ndarray, shape ``(action_horizon, action_dim)``
            Quantized DCT coefficients as returned by
            :meth:`_decode_dct_coefficients`.

        Returns
        -------
        np.ndarray, shape ``(action_horizon, action_dim)``
            Continuous action trajectory in the original value space
            (before any normalization that may be applied downstream).
        """
        return idct(
            dct_coeff / self._fast_tokenizer.scale, axis=0, norm='ortho'
        )


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        path = download.maybe_download(
            'gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}
        )
        with path.open('rb') as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read()
            )

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace('_', ' ')

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = (
            np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        )

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = ' '.join(map(str, discretized_state))
        prefix = f'Task: {cleaned_text}, State: {state_str};\n'
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError(
                'BinningTokenizer does not support encoding actions atm (only for inference use)'
            )
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(
            postfix_tokens
        )  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. '
                    'Consider increasing the `max_token_len` in your model config if this happens frequently.'
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )

    def extract_actions(
        self, tokens: np.ndarray, action_horizon: int, action_dim: int
    ) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if 'Action: ' not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(
                decoded_tokens.split('Action: ')[1].split('|')[0].strip()
            )
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape(
            [action_horizon, action_dim]
        )
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(
        self, tokens: np.ndarray | list[int]
    ) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return (
            self._paligemma_tokenizer.vocab_size()
            - 1
            - self._fast_skip_tokens
            - tokens
        )


class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    def __init__(
        self, max_len: int = 256, fsq_tokenizer_path: str | None = None
    ):
        self._max_len = max_len

        assert (
            fsq_tokenizer_path is not None
        ), 'fsq_tokenizer_path must be provided'
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split('/')[-1])
        base_path = tok_path.rsplit('/', 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                'params': ocp.StandardCheckpointHandler(),
                'opt_state': ocp.StandardCheckpointHandler(),
                'config': ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step,
                args=ocp.args.Composite(
                    config=ocp.args.JsonRestore(),
                    params=ocp.args.StandardRestore(),
                ),
            )
            config = restored['config']
            self._params = restored['params']
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f'Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}'
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply(
                {'params': params}, x, method=self._fsq_tokenizer.tokenize
            )
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply(
                {'params': params}, x, method=self._fsq_tokenizer.detokenize
            )
        )

        # Download base PaliGemma tokenizer
        path = download.maybe_download(
            'gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}
        )
        with path.open('rb') as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read()
            )

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace('_', ' ')

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = (
            np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        )

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = ' '.join(map(str, discretized_state))
        prefix = f'Task: {cleaned_text}, State: {state_str};\n'
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError(
                'FSQTokenizer does not support encoding actions atm (only for inference use)'
            )
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(
            postfix_tokens
        )  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. '
                    'Consider increasing the `max_token_len` in your model config if this happens frequently.'
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )

    def extract_actions(
        self, tokens: np.ndarray, action_horizon: int, action_dim: int
    ) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if 'Action: ' not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(
                decoded_tokens.split('Action: ')[1].split('|')[0].strip()
            )
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices('cpu')[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(
                    self._params, action_tokens[None, ...]
                )[0]
            return detok_act[: action_horizon * action_dim].reshape(
                [action_horizon, action_dim]
            )
        except Exception as e:
            logging.warning(f'Error decoding FSQ: {e}')
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(
        self, tokens: np.ndarray | list[int]
    ) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return (
            self._paligemma_tokenizer.vocab_size()
            - 1
            - self._fast_skip_tokens
            - tokens
        )
