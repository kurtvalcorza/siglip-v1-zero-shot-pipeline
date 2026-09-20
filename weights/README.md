# Base Model Weights Cache

This directory holds offline base-model weights, configurations, and cryptographic manifests for the SigLIP v1 zero-shot pipeline.

The model is organized in its own isolated subfolder corresponding to its canonical identifier:

```
weights/
└── siglip-base-patch16-256/
    ├── config.json
    ├── preprocessor_config.json
    ├── special_tokens_map.json
    ├── tokenizer.json
    ├── spiece.model
    ├── tokenizer_config.json
    ├── dimer-base-manifest.json
    ├── README.md (Model Card / Specification)
    ├── LICENSE
    └── model.safetensors (Excluded from Git; acquired via scripts/fetch_weights.py or DIMER upload)
```

## Available Base Model Snapshots

- [**`siglip-base-patch16-256`**](siglip-base-patch16-256/): Dedicated snapshot for Google's SigLIP (v1, 2023) vision-language model (`google/siglip-base-patch16-256`, 203,202,050 parameters).
  - [**Model Card & Specification**](siglip-base-patch16-256/README.md): Full technical architecture, zero-shot classification guidelines, embedding characteristics, and Apache-2.0 license terms.
  - [**Manifest**](siglip-base-patch16-256/dimer-base-manifest.json): Cryptographic record of byte counts and SHA-256 hashes for all snapshot files.

## DIMER Architecture & Git Tracking Strategy

In the DIMER workbench ecosystem:
1. **Large Binary Weights (`model.safetensors`):** The ~813 MB (812,856,640 bytes) weight payload is excluded from Git via `.gitignore` (`weights/**/*.safetensors`) and uploaded directly to DIMER as a model asset or downloaded using `scripts/fetch_weights.py`.
2. **Configuration & Tokenizers:** All accompanying configuration files (`config.json`, `preprocessor_config.json`, `special_tokens_map.json`), SentencePiece/tokenizer files (`tokenizer.json`, `spiece.model`, `tokenizer_config.json`), and cryptographic manifests are version-controlled in the repository so the pipeline and offline Docker containers can initialize vision processors and tokenizers without network dependencies.

## Management & Verification Tooling

Manage, download, and cryptographically verify model snapshots using [`scripts/fetch_weights.py`](../scripts/fetch_weights.py):

```bash
# Verify the existing snapshot in weights/siglip-base-patch16-256:
python scripts/fetch_weights.py --verify-only

# Download and verify default model into its dedicated subfolder:
python scripts/fetch_weights.py --dest weights/siglip-base-patch16-256

# Verify an explicit destination directory:
python scripts/fetch_weights.py --verify-only --dest weights/siglip-base-patch16-256
```
