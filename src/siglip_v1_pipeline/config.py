from __future__ import annotations

MODEL_ID = "google/siglip-base-patch16-256"
MODEL_REVISION = "b078df89e446d623010d890864d4207fe6399f61"
MODEL_FILENAME = "model.safetensors"
MODEL_SHA256 = "f0cee7c815135c44a515eff72ab3040499744920442bc25567cd04efc93f8f65"
MODEL_SIZE_BYTES = 812_856_640
MODEL_LICENSE = "Apache-2.0"

DEFAULT_MODEL_KEY = "siglip-base-patch16-256"
UNSAFE_WEIGHT_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".h5",
    ".msgpack",
)

ALLOWED_CHECKPOINT_FILES = (
    "config.json",
    MODEL_FILENAME,
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "spiece.model",
    "tokenizer_config.json",
)

DEFAULT_PROMPT_TEMPLATE = "This is a photo of {label}."
TEXT_MAX_LENGTH = 64
