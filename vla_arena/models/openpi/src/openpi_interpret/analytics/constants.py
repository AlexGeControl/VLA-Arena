"""Shared constants for the analytics pipeline.

Token ranges, modality definitions, and CMF pair specifications mirror
the backend constants but are self-contained to avoid import coupling.
"""

SAMPLED_LAYERS: list[int] = [0, 3, 6, 9, 12, 15, 17]
DEFAULT_CMF_LAYER: int = 17

NUM_HEADS: int = 8
HEAD_DIM: int = 256
NUM_ACTIONS: int = 50
NUM_SUFFIX: int = 51  # 1 state + 50 actions

TOKEN_RANGES: dict[str, tuple[int, int]] = {
    "base_0_rgb": (0, 256),
    "left_wrist_0_rgb": (256, 512),
    "right_wrist_0_rgb": (512, 768),
    "language": (768, 816),
    "state": (816, 817),
    "action": (817, 867),
}

VISUAL_RANGE: tuple[int, int] = (0, 768)
LANGUAGE_RANGE: tuple[int, int] = (768, 816)
STATE_RANGE: tuple[int, int] = (816, 817)

MODALITY_LABELS: list[str] = [
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "language",
    "state",
    "action",
]

CMF_PAIRS: dict[str, dict[str, object]] = {
    "S_to_L": {
        "query_suffix_indices": [0],
        "target_key_range": LANGUAGE_RANGE,
        "description": "Proprioceptive State → Language",
    },
    "S_to_V": {
        "query_suffix_indices": [0],
        "target_key_range": VISUAL_RANGE,
        "description": "Proprioceptive State → Visual",
    },
    "A_to_L": {
        "query_suffix_indices": list(range(1, 51)),
        "target_key_range": LANGUAGE_RANGE,
        "description": "Action Tokens → Language",
    },
    "A_to_V": {
        "query_suffix_indices": list(range(1, 51)),
        "target_key_range": VISUAL_RANGE,
        "description": "Action Tokens → Visual",
    },
    "A_to_S": {
        "query_suffix_indices": list(range(1, 51)),
        "target_key_range": STATE_RANGE,
        "description": "Action Tokens → Proprioceptive State",
    },
}
