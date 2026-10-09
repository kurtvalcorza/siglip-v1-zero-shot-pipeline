"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (six
modules, carried verbatim in dependency order), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

Generator /2 keys in use: ``modules`` lists every module of ``src/siglip_v1_pipeline/`` except
``__init__.py``; ``entry_module`` is ``config.py`` (it holds ``MODEL_ID``/``MODEL_REVISION``/
``MODEL_LICENSE`` and the model key under the package's own spelling ``DEFAULT_MODEL_KEY``, mapped by
``identity_names``); ``rewrites`` carries two rules — the fleet ``DEFAULT_WEIGHTS_DIR`` rule and the
``__file__`` use inside ``model.resolve_weights_path`` (a repository-checkout convenience that a
standalone notebook has no checkout for); ``model_load`` lets the pipeline pick CUDA when it is visible
(the fine-tuning stage is where that matters; CPU is the documented fallback).

This template configures an E2E zero-shot-classification fine-tuning workflow: the pinned
google/siglip-base-patch16-256 snapshot is digest-verified and loaded, 360 CC0 iNaturalist photographs of
six bird species are fetched with per-file digests, validated and split by photograph, the drawn synthetic
shapes are classified through the inference contract, the frozen model's zero-shot accuracy / macro F1 /
text-to-image mAP over the held-out photographs is measured beside two non-neural baselines, a bounded
fine-tuning of the vision tower's last blocks runs in the kernel with SigLIP's sigmoid loss, the held-out
split is scored again per species, the adapted model re-scores the shapes, and the adapter is exported and
reloaded.

2026-10-05 review fix cycle (SIG-M1..M4, SIG-m1): generator /2.2 keys ``isolated_runtime`` (nothing is installed into
the kernel; a hash-locked uv environment runs every later cell, so Run all needs no restart), ``infrastructure_labels``
and ``guided`` (audience, how-to-use, roadmap, task contract; predictions, worked checkpoints, a change-one-thing
experiment, troubleshooting, glossary and a conclusion template). Quality outcomes are reported as verdicts instead of
asserted; ``SiglipPipeline.adapt`` restarts from the pinned base on every call; BYOD takes a path or one uploaded zip.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "siglip_v1_pipeline",
    "repo_name": "siglip-v1-zero-shot-pipeline",
    "stem": "siglip_v1_zero_shot",
    "notebook_name": "siglip_v1_zero_shot_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline): managed CPython, a size- and
    # SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "SiglipPipeline",
    "weights_key": "siglip-base-patch16-256",
    "modules": ["config.py", "model.py", "metrics.py", "samples.py", "pipeline.py", "provenance.py"],
    "entry_module": "config.py",
    "identity_names": {"MODEL_KEY": "DEFAULT_MODEL_KEY"},
    "rewrites": [
        ["^DEFAULT_WEIGHTS_DIR = Path\\(__file__\\)[^\\n]*$", 'DEFAULT_WEIGHTS_DIR = Path.cwd() / "weights" / DEFAULT_MODEL_KEY  # standalone rewrite (build_notebook.py): working-directory snapshot, no repository checkout'],
        ["^    repo_root = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]$", "    repo_root = Path.cwd()  # standalone rewrite (build_notebook.py): no repository checkout to resolve"],
    ],
    "model_load": "SiglipPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)",
    "runtime_imports": ["torch", "transformers", "numpy"],
    "title": "SigLIP v1 Zero-Shot Pipeline — DIMER E2E zero-shot classification fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/siglip-v1-zero-shot-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/siglip-v1-zero-shot-pipeline/blob/main/tutorials/siglip_v1_zero_shot_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fsiglip--base--patch16--256-ffcc4d?style=flat",
            "https://huggingface.co/google/siglip-base-patch16-256",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-huggingface%2Ftransformers-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/huggingface/transformers",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2303.15343-b31b1b.svg", "https://arxiv.org/abs/2303.15343"),
    ],
    "capability": "zero-shot image classification, image/text embeddings, cosine similarity, text-to-image retrieval and bounded supervised fine-tuning of the vision tower's last blocks on a labelled-photograph dataset, using the pinned `google/siglip-base-patch16-256` weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned `google/siglip-base-patch16-256` snapshot (an 813 MB `model.safetensors`; no pickle is opened anywhere), fetches "
        "the 360 pinned iNaturalist photographs from the project's open-data bucket (about 39 MB, each refused on any byte-size "
        "or SHA-256 mismatch), cuts them per species into 216 / 48 / 96 training, validation and test photographs, classifies "
        "three drawn shapes through the inference contract with an input manifest and a rejection probe, measures the frozen "
        "model's zero-shot accuracy, macro F1 and text-to-image mAP over the 96 test photographs beside the majority-floor and "
        "colour-nearest-neighbour baselines, runs a bounded fine-tuning of the vision tower's last two blocks and attention-pool "
        "head with SigLIP's sigmoid loss and validation-accuracy epoch selection, scores the held-out photographs again per species, "
        "re-scores the drawn shapes with the adapted model, exports the adapter as safetensors with a manifest, and reloads that "
        "artifact into a fresh pipeline to verify embedding parity. The default path needs no repository clone, no DIMER worker "
        "or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.2 §5). On CPU the whole path "
        "took about three and a half minutes of model time on the build workstation's CPU after the downloads (expect longer on a 2-vCPU hosted runtime); a CUDA runtime is used automatically when present and finishes in "
        "a few minutes."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to a zip or folder "
        "in the runtime (Colab, Kaggle or Jupyter) or leave it empty to upload one zip in Colab, then choose **Run after** from "
        "that cell. The zip or folder holds a `labels.csv` (columns `id`, `file`, `label`) beside the image files, the label text "
        "being what the prompt names. After the split every label needs at least one training, validation and test photograph and "
        "the training split eight, so the **effective minimum is 12 photographs for two labels** (6 per label), 15 for three, "
        "and 4 per label from four labels upward; with so few, each label's held-out score rests on one or two photographs and "
        "the cell says so. They pass through the same validation, seeded stratified "
        "split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as the iNaturalist sample. "
        "The expected schema and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside "
        "this runtime. BYOD is optional and never part of the default path."
    ),
    "intro": (
        "`google/siglip-base-patch16-256` is the base SigLIP model of Zhai, Mustafa, Kolesnikov and Beyer (ICCV 2023) — a ViT-B/16 image tower at 256×256 "
        "with an attention-pool head and a 12-layer text tower over a 32,000-piece English SentencePiece vocabulary, trained on the English pairs of WebLI with the sigmoid loss; 203,202,050 parameters, "
        "published under the **Apache-2.0** licence. A photograph and a prompt are scored by the cosine of their projected "
        "embeddings, scaled and shifted by the model's learned `logit_scale` and `logit_bias` and read through a sigmoid: the "
        "scores are independent per pair, **not calibrated probabilities**, **prompt/label dependent**, and the model never "
        "abstains — the highest-scoring label is returned whatever the image shows.\n\n"
        "What this notebook adds to inference is **adaptation with labelled photographs**. The dataset is real and where the "
        "frozen model has room to improve: 360 CC0-licensed, research-grade iNaturalist photographs of six common North "
        "American birds (**CC0 1.0**; four small sparrows, a junco and two finches, 60 per species, one per observer), pinned "
        "by photo id, byte size and SHA-256 and fetched from the project's open-data bucket at run time. Zero-shot prompts from "
        "the common names separate these species only partly (the build record measured accuracy 0.76 frozen on the 96 test "
        "photographs, with the Chipping Sparrow at 0.44 recall and the White-throated Sparrow's average precision at 0.34), so the honest question is narrow: does a bounded "
        "fine-tuning of the vision tower's last blocks on 216 photographs move zero-shot accuracy and the retrieval view "
        "(**text-to-image mAP**) on an image-disjoint test split, per species, against two **non-neural baselines** (the "
        "**majority floor** and a **colour nearest neighbour**)? Nothing here is a quality claim about your photographs: it is "
        "one seeded split of one sample.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (an 8-file manifest) — no pickle is opened anywhere "
        "in this notebook. Section 3 stages and digest-verifies those files before the processor or the model is constructed."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter and has met image classification, and wants to see how a vision-language model classifies photographs of classes it was never trained on, how to measure that honestly, and how to adapt it to a small labelled set without fooling themselves. No prior experience with SigLIP, CLIP or fine-tuning is assumed; each term is explained where it first matters and again in the **Glossary** at the end. A T4 GPU runtime finishes the fine-tuning in a few minutes; CPU works but is several times slower.\n\n**Input → Model → Output.**\n\n| | Zero-shot classification | Text-to-image retrieval | Bounded fine-tuning |\n|---|---|---|---|\n| Input | one photograph and candidate labels, each put into a prompt | a text query and a gallery of photographs | labelled photographs (216 training and 48 validation in the sample) |\n| Model | image tower and text tower; the cosine of the two embeddings × `logit_scale` + `logit_bias`, read through a sigmoid | the same two towers; cosine ranking | the last two vision-tower blocks, the post-layernorm and the attention-pool head, trained with the sigmoid loss; everything else frozen |\n| Output | one independent sigmoid score per label, ranked (never an abstention) | gallery photographs ranked by cosine | an 85 MB safetensors adapter, and held-out accuracy, macro F1 and text-to-image mAP |\n\n**How to use this notebook.** Choose a runtime (**Runtime → Change runtime type → T4 GPU** is faster; CPU works), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer from the recorded run (the Kaggle T4 run of 20 September 2026; a CPU run can differ in the last digit). Section 10 is a **change-one-thing experiment**, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 photographs, validation and a leakage-free split → 5 the inference contract on three drawn shapes → 6 two baselines and the frozen model *(evaluation practice)* → 7 bounded fine-tuning → 8 held-out evaluation, per species → 9 the shapes again, export and reload → 10 change one thing (optional) → conclude. Sections 5 and 7 carry the **core concepts** (sigmoid scores, the sigmoid loss, what is trained), Sections 4, 6 and 8 the **evaluation practice** (baselines, leakage, held-out reading) and Sections 1–3 and 9 the **engineering** (environment, provenance, export)."
            )
        ]
    },
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled photograph set, validate it and split it per species without "
        "leakage; classify drawn shapes through the public API and read sigmoid scores correctly (independent, uncalibrated, no "
        "abstention); measure the frozen model's zero-shot accuracy, macro F1 and text-to-image mAP beside two non-neural "
        "baselines and read the per-species breakdown; run a bounded fine-tuning with SigLIP's sigmoid loss, explicit "
        "hyperparameters and validation-based epoch selection; evaluate on an image-disjoint test split with two prompt sets; "
        "re-score drawings from a different image family with the adapted model; and export a safetensors adapter that "
        "reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "object detection, semantic segmentation, OCR, caption generation, calibrated probabilities or universal thresholds, "
        "fine-tuning of the text tower, the embeddings, `logit_scale` or `logit_bias`, training on photographs that are not the "
        "pinned sample or your own uploads, evaluation on iNaturalist or any benchmark proper (only one seeded 360-photograph "
        "sample is scored here), and any claim that six bird species stand in for your classes. The repository exposes none of "
        "these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with SigLIP, CLIP or fine-tuning. The notebook explains sigmoid scores, `logit_scale`/`logit_bias`, prompts, embeddings, accuracy, macro F1, average precision and text-to-image mAP, the attention-pool head and the adapter where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is slow but adequate: the build record measured about 7 s to embed and score the 96 test photographs and 181 s for the six epochs of fine-tuning (216 photographs per epoch through the full vision tower, the last two blocks and the head training), including the per-epoch validation scoring, so the whole default path is about three and a half minutes of model time on the build workstation's CPU with the snapshot and photographs already cached (a 2-vCPU hosted runtime will be several times slower); a hosted T4 finishes it in a few minutes. The pinned `torch==2.14.0` install and the 813 MB checkpoint are the large downloads of the run; the photographs add about 39 MB.",
        "- **Knowledge:** basic Python and PIL; what a sigmoid score and a cosine similarity are; what accuracy, macro F1 and average precision measure and why none is a human judgement; why a high score is not a correct label.",
        "- **Data contract:** records are `{id, image, label}` — a PIL image or a file decodable by Pillow with sides up to `MAX_IMAGE_SIDE` (4096) px and a label of at most 64 plain characters (the prompt is `DEFAULT_PROMPT_TEMPLATE` with the label, or its display name, filled in). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..20,000 records over 2..100 labels; splitting is stratified per label after pixel-digest de-duplication so no photograph lands in two splits. BYOD accepts one zip (or folder) of images plus a `labels.csv` in that shape; after the split every label needs a training, a validation and a test photograph and the training split eight, so the effective BYOD minimum is **12 photographs for two labels**, 15 for three and 4 per label from four labels (`min_byod_photographs(n_labels)` computes it). Every refusal names the `labels.csv` line, the file and the rule.",
        "- **Validation is structural, not semantic:** every image is opened and decoded and every label checked, but nothing checks that a label is right — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path fetches 360 JPEG/PNG files from `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/medium.<ext>` (about 39 MB in total), each pinned by byte size and SHA-256 in the carried `samples.py` and refused on any mismatch; every photograph's iNaturalist observation page and observer login are kept in its record. Each photograph carries the CC0 1.0 licence its observer chose (Gurari-style redistribution is not needed: nothing is committed to the repository).",
    ],
    "cells": [
        {
            "md": (
                "## 4. iNaturalist photographs and split\n\n"
                "`fetch_corpus` returns the 360 pinned photographs from the cache under `weights/inat-birds/` or the "
                "iNaturalist open-data bucket — every cached file is re-hashed and every fetched file refused on any byte-size or "
                "SHA-256 mismatch — and `read_corpus` decodes them into `{{id, image, label}}` records with their observation "
                "page, observer and species names. `build_sample_dataset` draws a seeded stratified split per species (36 / 8 / "
                "16 → 216 / 48 / 96). `validate_dataset` then checks every record against the contract, `check_split_disjoint` "
                "asserts no photograph (by decoded-pixel digest) is shared, `observer_overlap` reports how many observers "
                "contributed to more than one split (an observation about the draw, not an assertion), and the training split's "
                "labels table is written to `outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 360 photographs, the six species with 36 / 8 / 16 each, three digests, and four refusal probes — a "
                "duplicate id, an image over the side ceiling, a dataset with one label and one too small to split — each "
                "rejected before the model does anything.\n\n"
                "*Evaluation practice.* **Bring your own data (optional):** set `USE_BYOD = True` and either `BYOD_PATH` (a zip or a "
                "folder holding `labels.csv` and the images, as a path in this runtime — this works on Colab, Kaggle and Jupyter) or "
                "leave `BYOD_PATH` empty to upload exactly one zip through the Colab dialog; then choose **Run after** from this "
                "cell. The effective minimum is 12 photographs for two labels (see the Prerequisites); the cell reports how many "
                "duplicate images the split dropped and warns when a label has fewer than five held-out test photographs.\n\n"
                "**Predict before running:** the split is made per photograph, and each of the 360 photographs comes from a "
                "different iNaturalist observation. Will any *observer* (photographer) contribute photographs to more than one "
                "split — and if so, is that leakage?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n"
                "from collections import Counter\n"
                "from dataclasses import asdict\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_zip = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_zip.exists():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding labels.csv and the image files.')\n"
                "        file_name = byod_zip.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{file_name}}: upload one .zip holding labels.csv and the image files.')\n"
                "        byod_zip = Path('work') / 'byod.zip'\n"
                "        byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    byod_minimum = min_byod_photographs(len({{r['label'] for r in records}})) if len({{r['label'] for r in records}}) >= MIN_CLASSES else None\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    display_names = {{}}\n"
                "    raw_rows = {{'byod': len(records), 'duplicate_images_dropped': len(records) - sum(len(part) for part in splits.values()), 'effective_minimum': byod_minimum}}\n"
                "    thin = {{label: n for label, n in sorted(Counter(r['label'] for r in splits['test']).items()) if n < 5}}\n"
                "    if thin:\n"
                "        print({{'caution': 'fewer than five held-out test photographs for some labels: their scores move in steps of 1/n and carry no dispersion estimate; add photographs before reading per-label numbers', 'test_photographs': thin}})\n"
                "else:\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/inat-birds')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "    display_names = {{key: common for key, (_scientific, common) in SPECIES.items()}}\n"
                "    raw_rows = {{'photographs': len(corpus), 'bytes': sum(len(v) for v in corpus_files.values()), 'observers': len({{r['observer'] for r in corpus}})}}\n"
                "# The training split must hold MIN_RECORDS; validation and test only need a photograph per label (split_dataset checks that).\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "classes = class_names(train_records)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'observer_overlap': observer_overlap(splits), 'classes': classes}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'label_counts': manifest['label_counts'], 'image_side': manifest['image_side'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{'id': example['id'], 'label': example['label'], 'display_name': display_names.get(example['label'], example['label']), 'size': list(example['image'].size), 'observation': example.get('inat_observation_url')}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'image over the side ceiling': [{{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8))}}, *train_records[1:8]],\n"
                "    'one label only': [{{**r, 'label': 'bird'}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the three split sizes and their digests, the four refusals (each names its rule), and the "
                "`observer_overlap` line.\n\n"
                "<details><summary>Check your reasoning</summary>Yes: in the recorded run 53 of the 222 observers contributed "
                "photographs to more than one split. No photograph is shared (the pixel-digest check proves that), but the same "
                "photographer's camera, site and style can appear in training and test, so the test score may be a little "
                "optimistic for photographs from people the model has never seen. The notebook reports the overlap instead of "
                "hiding it; on your own data, split by photographer or session when your images come from few sources.</details>"
            ),
        },
        {
            "md": (
                "## 5. Classify through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: three 32×32 synthetic shapes "
                "— a red square, a green circle and a blue triangle — rendered in code as ASCII PPM files exactly as the "
                "repository's `examples/sample-data/generate_samples.py` renders them and digest-asserted against "
                "`SHA256SUMS`, a different image family from the photographs, and images the model will be asked to classify "
                "again after adaptation. `validate_inputs` applies exactly the checks the public operations apply and returns an "
                "input manifest; a remote URL is validated too and its rejection recorded as a finding. `zero_shot_classify` "
                "returns one sigmoid score per candidate label, **ordered by descending score**; `retrieve` ranks the gallery by "
                "cosine to a query. The per-grid `evaluation_report` on three drawn shapes is `sample-sanity` — plumbing "
                "evidence, not a measurement; whether the classifier is *right* is what Section 6 measures on 96 photographs.\n\n"
                "*Core concept.* A **sigmoid score** is the model's own logit for one image–prompt pair — the cosine of the two "
                "embeddings times `logit_scale`, plus `logit_bias` — squashed into 0..1. Each pair is scored on its own.\n\n"
                "**Predict before running:** each drawn shape is scored against four candidate labels. Will each image's four "
                "scores add up to 1?"
            ),
            "code": (
                "SAMPLE_DIGESTS = {{  # examples/sample-data/SHA256SUMS\n"
                "    'red_square.ppm': 'b38ff0c9131677ed6cf09832eff40a22841e1a4725d426fad3b1bf6a1dbdb096',\n"
                "    'green_circle.ppm': '2e3e657686a0f6a6df3f3621d21a410d20d9a47b109f06f9405faa4f747a663f',\n"
                "    'blue_triangle.ppm': 'f0f4c38c7af92b3a6b1d55a25272156edd87b41e029e116cfa059003dae029a3',\n"
                "}}\n"
                "WIDTH, HEIGHT, BACKGROUND = 32, 32, (245, 245, 245)\n\n\n"
                "def shape_mask(shape, x, y):\n"
                "    if shape == 'square':\n"
                "        return 8 <= x < 24 and 8 <= y < 24\n"
                "    if shape == 'circle':\n"
                "        return (x - 16) ** 2 + (y - 16) ** 2 <= 9**2\n"
                "    if not 7 <= y < 26:\n"
                "        return False\n"
                "    half = (y - 7) // 2\n"
                "    return 16 - half <= x <= 16 + half\n\n\n"
                "def render_ppm(foreground, shape):\n"
                "    # The repository's generate_samples.py rendering: ASCII P3, 24 values per line.\n"
                "    lines = ['P3', f'{{WIDTH}} {{HEIGHT}}', '255']\n"
                "    for y in range(HEIGHT):\n"
                "        row = []\n"
                "        for x in range(WIDTH):\n"
                "            pixel = foreground if shape_mask(shape, x, y) else BACKGROUND\n"
                "            row.extend(str(v) for v in pixel)\n"
                "        for start in range(0, len(row), 24):\n"
                "            lines.append(' '.join(row[start : start + 24]))\n"
                "    return '\\n'.join(lines) + '\\n'\n\n\n"
                "Path('outputs/sample-data').mkdir(parents=True, exist_ok=True)\n"
                "specs = {{'red_square.ppm': ((220, 40, 40), 'square'), 'green_circle.ppm': ((40, 170, 75), 'circle'), 'blue_triangle.ppm': ((40, 90, 220), 'triangle')}}\n"
                "shape_images = []\n"
                "for name, (foreground, shape) in specs.items():\n"
                "    path = Path('outputs/sample-data') / name\n"
                "    path.write_bytes(render_ppm(foreground, shape).encode('ascii'))\n"
                "    digest = hashlib.sha256(path.read_bytes()).hexdigest()\n"
                "    if digest != SAMPLE_DIGESTS[name]:\n"
                "        raise ValueError(f'Synthetic sample digest mismatch for {{name}}: {{digest}} != {{SAMPLE_DIGESTS[name]}}')\n"
                "    shape_images.append(path)\n"
                "shape_labels = ['red square', 'green circle', 'blue triangle']\n"
                "candidate_labels = [*shape_labels, 'abstract geometric shape']\n"
                "shape_sha256 = {{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shape_images}}\n"
                "print({{'ceilings': {{'TEXT_MAX_LENGTH': TEXT_MAX_LENGTH, 'DEFAULT_PROMPT_TEMPLATE': DEFAULT_PROMPT_TEMPLATE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'MIN_CLASSES': MIN_CLASSES, 'image_contract': '256x256 RGB after processor resize', 'device': str(pipe.device)}}}})\n"
                "input_manifest = validate_inputs(shape_images, candidate_labels, top_k=len(shape_images), names=[p.name for p in shape_images])\n"
                "try:\n"
                "    validate_inputs('https://example.invalid/not-allowed.png', candidate_labels)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'remote-url-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'shapes': [p.name for p in shape_images], 'sha256': {{k: v[:16] + '...' for k, v in shape_sha256.items()}}, 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                "classifications, retrievals, classification_rows = [], [], []\n"
                "t0 = time.perf_counter()\n"
                "for image_path, expected in zip(shape_images, shape_labels, strict=True):\n"
                "    scores = pipe.zero_shot_classify(image_path, candidate_labels)\n"
                "    classifications.append(scores)\n"
                "    classification_rows.append({{'image': image_path.name, 'expected_label': expected, 'predicted_label': scores[0].label, 'scores': [asdict(x) for x in scores]}})\n"
                "    print(image_path.name, '->', [(s.label, round(s.score, 4)) for s in scores])\n"
                "for query in shape_labels:\n"
                "    retrievals.append(pipe.retrieve(query, shape_images, top_k=len(shape_images)))\n"
                "shape_embeddings = pipe.embed_image(shape_images)\n"
                "checks = {{\n"
                "    'one_ranking_per_image': len(classifications) == len(shape_images),\n"
                "    'scores_in_unit_interval': all(0.0 <= s.score <= 1.0 for ranking in classifications for s in ranking),\n"
                "    'rankings_descending': all(ranking[i].score >= ranking[i + 1].score for ranking in classifications for i in range(len(ranking) - 1)),\n"
                "    'embeddings_unit_norm': bool(np.allclose(np.linalg.norm(shape_embeddings, axis=1), 1.0, atol=1e-4)),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'inference output failed a sanity check: {{checks}}')\n"
                "result = {{'classifications': classifications, 'retrievals': retrievals, 'gallery_ids': [p.name for p in shape_images], 'embedding_shapes': {{'image': list(shape_embeddings.shape)}}}}\n"
                "targets = {{'labels': shape_labels, 'retrieval_indices': list(range(len(shape_images)))}}\n"
                "frozen_scene = evaluation_report(result, targets, sample_kind='synthetic')\n"
                "print({{'checks': checks, 'seconds': round(time.perf_counter() - t0, 2), 'frozen_scene': {{m['id']: round(m['value'], 3) for m in frozen_scene['metrics']}}, 'verdict': frozen_scene['verdict']}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the rankings are ordered by score, every score lies in 0..1, and the remote URL was refused "
                "and recorded as a finding instead of being fetched.\n\n"
                "<details><summary>Check your reasoning</summary>No. SigLIP scores every image–label pair independently through "
                "a sigmoid, so the four scores need not sum to 1 — they can all be small, or several can be high. That is why a "
                "score is not a probability that the label is right, and why the model never abstains: the highest-scoring "
                "label is returned even when none fits. A softmax model (CLIP) would force the scores to sum to 1 instead.</details>"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's zero-shot score on the test photographs\n\n"
                "Three systems frame the adaptation, each read three ways. The **majority floor** answers every photograph with "
                "the most frequent training label (accuracy 1/6 on a balanced split, chance-level macro F1). The **colour "
                "nearest neighbour** answers with the label of the training photograph whose 3×3 mean-colour grid is closest — "
                "a classifier that knows the image through 27 numbers. The **frozen model** is scored by `pipe.evaluate`: one "
                "prompt per species (`DEFAULT_PROMPT_TEMPLATE` with the common name), the model's own sigmoid-scaled logits as "
                "the score grid, **accuracy** and **macro F1** of the top prompt, the per-species recall, and **text-to-image "
                "mAP** — each prompt as a query ranking all 96 photographs, the average precision of its own species, averaged "
                "over the six — the retrieval view of the same scores and the smoother of the three on a small set. A second "
                "prompt set built from the scientific names is scored too, to show how much the number is the prompt's. Expect "
                "the frozen model far above both baselines — it is a trained zero-shot classifier — and read the per-species "
                "breakdown: the build record measured accuracy 0.76 / macro F1 0.76 / mAP 0.72 frozen, with the Chipping Sparrow "
                "at 0.44 recall and the American Goldfinch at 0.94, and the White-throated Sparrow's recall of 0.81 resting on an average precision of 0.34 — it is over-predicted, not missed.\n\n"
                "*Evaluation practice.* The cell reports a **verdict** — whether the frozen model is above both baselines — and "
                "records it; it does not stop the notebook when the answer is no (on your own data that answer is a finding).\n\n"
                "**Predict before running:** write down the test accuracy you expect from the majority floor, the colour nearest "
                "neighbour and the frozen model on six balanced species — and which of the three you expect to be closest to chance."
            ),
            "code": (
                "baseline_majority = majority_baseline(train_records, test_records, classes)\n"
                "baseline_neighbour = colour_neighbour_baseline(train_records, test_records, classes)\n"
                "METRICS = ('accuracy', 'macro_f1', 't2i_map')\n"
                "print({{'majority_baseline': {{k: round(baseline_majority[k], 3) for k in METRICS}}, 'n': baseline_majority['n'], 'note': baseline_majority['baseline']}})\n"
                "print({{'colour_neighbour_baseline': {{k: round(baseline_neighbour[k], 3) for k in METRICS}}, 'note': baseline_neighbour['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, classes=classes, class_names_map=display_names)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'prompt_template': frozen_test['prompt_template'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "frozen_fields = {{c: {{'n': v['n'], 'recall': round(v['recall'], 2), 'ap': round(v['ap'], 2)}} for c, v in frozen_test['per_class'].items()}}\n"
                "print({{'by_species_frozen': frozen_fields}})\n"
                "scientific_names = {{key: scientific for key, (scientific, _common) in SPECIES.items()}} if not USE_BYOD else {{}}\n"
                "frozen_scientific = pipe.evaluate(test_records, classes=classes, class_names_map=scientific_names, prompt_template='This is a photo of {{label}}.')\n"
                "print({{'frozen_model_test_scientific_name_prompts': {{k: round(frozen_scientific[k], 3) for k in METRICS}}}})\n"
                "# A reported verdict, not an assertion: on your own data the frozen model may not beat a baseline, and that is a finding.\n"
                "frozen_beats = {{'t2i_map_above_majority_floor': bool(frozen_test['t2i_map'] > baseline_majority['t2i_map']), 'accuracy_above_colour_neighbour': bool(frozen_test['accuracy'] > baseline_neighbour['accuracy'])}}\n"
                "frozen_verdict = 'above both baselines' if all(frozen_beats.values()) else 'not above both baselines: read the per-class breakdown and the prompts before adapting'\n"
                "print({{'frozen_vs_baselines': frozen_verdict, 'checks': frozen_beats}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the three systems on three metrics, the per-species recall and average precision, and how "
                "far the scientific-name prompts move the frozen score.\n\n"
                "<details><summary>Check your reasoning</summary>In the recorded run the majority floor scored accuracy 0.167 "
                "(1/6, macro F1 0.048) — chance by construction — and the colour nearest neighbour 0.26 (macro F1 0.261), barely "
                "above it: four of the six species are streaked brown sparrows, so 27 colour numbers say little. The frozen model "
                "scored accuracy 0.76, macro F1 0.76 and text-to-image mAP 0.72 with no training at all, and the verdict was "
                "*above both baselines*. The same model with scientific-name prompts scored about 0.54: the number belongs to "
                "the prompt as much as to the model.</details>"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the vision tower's last blocks\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_VISION_LAYERS` blocks of the vision tower, its post-layernorm and "
                "its attention-pool head — two blocks by default, 21,264,384 of 203,202,050 parameters; the text tower, the "
                "embeddings, `logit_scale` and `logit_bias` stay frozen. The six class prompts are embedded once by the frozen "
                "text tower; every batch of photographs is run through the vision tower, scored against the prompts with the "
                "model's own sigmoid-scaled logits, and trained with SigLIP's pairwise sigmoid loss (+1 for the gold species, "
                "−1 for the other five). AdamW at a fixed learning rate, gradient clipping at 1.0, seeded shuffling and no "
                "scheduler. Epoch 0 records the frozen model's validation metrics; every epoch is scored on the 48 validation "
                "photographs, and the epoch with the highest validation accuracy is kept (the first one on a tie). Accuracy on 48 "
                "photographs moves in steps of about 2 %, so the choice is coarse; validation text-to-image mAP is printed beside it "
                "as a second view, and the 96-photograph test split is what the numbers are read from.\n\n"
                "Watch the training loss fall from about 1.8 to below 0.1 within six epochs while the validation mAP peaks "
                "early: 216 photographs are few, the last blocks memorise them, and the selector's job is to stop before that "
                "hurts. The build record kept epoch 4 of six (validation mAP rose from 0.68 frozen to 0.90 there); the default is the "
                "configuration that gained on the held-out split.\n\n"
                "*Core concept.* Every call to `pipe.adapt` starts from the **pinned base**: tensors an earlier call changed are "
                "restored first, so epoch 0 is always the frozen model. Re-running this cell with a changed field is therefore a "
                "fresh run, not continued training — but it replaces the default results that Sections 8 and 9 export. To compare "
                "a change side by side with the default and leave the default exports untouched, use Section 10.\n\n"
                "**Predict before running:** the training loss will fall every epoch. Will the validation mAP keep rising with "
                "it, and will the last epoch be the one that is kept?"
            ),
            "code": (
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_VISION_LAYERS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "settings = {{'epochs': EPOCHS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'trainable_vision_layers': TRAINABLE_VISION_LAYERS}}\n"
                "if settings != {{'epochs': 6, 'lr': 5e-5, 'batch_size': 16, 'trainable_vision_layers': 2}}:\n"
                "    print({{'note': 'changed settings: this run starts again from the pinned base and replaces the default results of Sections 8-9; Section 10 compares a change side by side instead', 'settings': settings}})\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_vision_layers=TRAINABLE_VISION_LAYERS, class_names_map=display_names, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'classes': adapt_result['classes'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "**What to notice:** epoch 0 (the frozen model, before any update), the loss column, the validation columns, "
                "and which epoch is reported as `best_epoch`.\n\n"
                "<details><summary>Check your reasoning</summary>No to both. In the recorded run the loss fell from about 1.8 to "
                "below 0.1 within six epochs, while the validation numbers rose from the frozen model's (validation mAP 0.68) and "
                "then stopped improving: epoch 4 of six was kept on validation accuracy (validation mAP 0.90 at that epoch). 216 photographs are few and the last "
                "blocks start to memorise them, so a falling training loss is not evidence of a better classifier; the held-out "
                "split decides, and the test split is kept for Section 8.</details>"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test photographs were never used for training or epoch selection, and no photograph appears in two splits. "
                "The adapted model is scored exactly as the frozen model was in Section 6 — the same six prompts — the four "
                "systems are put side by side on the three metrics, the per-species breakdown is repeated, and the "
                "scientific-name prompt set is scored again (the vision tower was adapted, not the prompts, so a gain that "
                "carries to a prompt set it never saw is the more general one). Read it in this order: **text-to-image mAP** "
                "first (the metric the epoch was selected on — the build record measured 0.72 → 0.89), then accuracy and macro "
                "F1 (0.76 → 0.84 and 0.76 → 0.85, eight photographs of 96), then the per-species recall, where the "
                "Chipping Sparrow moved from 0.44 to 0.69 and the Song Sparrow from 0.75 to 0.94 while the White-throated Sparrow's recall fell from 0.81 to 0.75 as its average precision rose from 0.34 to 0.87 — it stopped being the default answer. The cell reports a "
                "**verdict** — `improved`, `no gain` or `worse` on the held-out mAP — and records it in the evaluation report; a "
                "fine-tune that does not help is a finding, not an error, and Section 9 still exports, reloads and writes the "
                "provenance. Ninety-six photographs from one seeded split give **no dispersion "
                "estimate**; the deltas are sample-sanity evidence that the adaptation contract works, not a benchmark, and a "
                "gain on six birds says nothing about your classes until you measure them.\n\n"
                "**Predict before running:** will the fine-tune help every species equally? Which species do you expect to gain "
                "most — one the frozen model already got right, or one it confused?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, classes=classes, class_names_map=display_names)\n"
                "adapted_val = pipe.evaluate(val_records, classes=classes, class_names_map=display_names)\n"
                "adapted_fields = {{c: {{'n': v['n'], 'recall': round(v['recall'], 2), 'ap': round(v['ap'], 2)}} for c, v in adapted_test['per_class'].items()}}\n"
                "adapted_scientific = pipe.evaluate(test_records, classes=classes, class_names_map=scientific_names, prompt_template='This is a photo of {{label}}.')\n"
                "comparison = {{metric: {{'majority': round(baseline_majority[metric], 3), 'neighbour': round(baseline_neighbour[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['scientific_name_prompts'] = {{metric: {{'frozen': round(frozen_scientific[metric], 3), 'adapted': round(adapted_scientific[metric], 3)}} for metric in METRICS}}\n"
                "comparison['by_species'] = {{c: {{'n': frozen_fields[c]['n'], 'frozen_recall': frozen_fields[c]['recall'], 'adapted_recall': adapted_fields[c]['recall'], 'frozen_ap': frozen_fields[c]['ap'], 'adapted_ap': adapted_fields[c]['ap']}} for c in classes}}\n"
                "# Reported verdicts, not assertions: a non-improving fine-tune is a result to record, and export and reload still run.\n"
                "delta_map = adapted_test['t2i_map'] - frozen_test['t2i_map']\n"
                "adaptation_verdict = 'improved' if delta_map > 0 else ('no gain' if delta_map == 0 else 'worse')\n"
                "comparison['verdicts'] = {{'frozen_vs_baselines': frozen_verdict, 'adapted_vs_frozen_t2i_map': adaptation_verdict}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': DEFAULT_MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'classes': classes,\n"
                "    'display_names': display_names,\n"
                "    'baselines': {{'majority': baseline_majority, 'colour_neighbour': baseline_neighbour}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'frozen_test_scientific_names': frozen_scientific,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'test_metrics_scientific_names': adapted_scientific,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'adaptation_verdict': adaptation_verdict, 'test_t2i_map_delta': round(delta_map, 3), 'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the four-way table, the `verdicts` row, and the per-species rows where recall and average "
                "precision move in different directions.\n\n"
                "<details><summary>Check your reasoning</summary>Not equally. In the recorded run the verdict was *improved*: "
                "test mAP 0.718 → 0.891 and accuracy 0.760 → 0.844 (eight photographs of 96). The largest gains were on species "
                "the prompts separated worst — the Chipping Sparrow's recall rose from 0.44 to 0.69 and the Song Sparrow's from "
                "0.75 to 0.94 — while the White-throated Sparrow's recall *fell* from 0.81 to 0.75 as its average precision rose "
                "from 0.34 to 0.87: the frozen model had been answering it for photographs of other sparrows. One seeded split of "
                "96 photographs gives no dispersion estimate, so read these as evidence that the adaptation contract works, not "
                "as a benchmark.</details>"
            ),
        },
        {
            "md": (
                "## 9. Re-score the drawn shapes, export the adapter and reload it\n\n"
                "The three shapes from Section 5 are classified again by the adapted model — drawings, a different image family "
                "from the photographs it was tuned on, so this is a small look at what the adaptation did *outside* its corpus "
                "(the build record's rankings are in `docs/release-verification.md`; a changed ranking here is a finding to record, not a "
                "failure) — and reported with the per-grid `evaluation_report` (`sample-sanity`). Both score sets are written "
                "as JSON.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the vision tower's last two blocks, post-layernorm and "
                "attention-pool head, about 85 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact "
                "format, the base model id and revision, the digest of the base `model.safetensors`, the classes and prompt "
                "template, the tensor names, the file size and SHA-256, the training configuration and the epoch history "
                "(OUT8). `SiglipPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest, its digest "
                "and its exact tensor set **before** deserialising, refuses any tensor outside the vision tower, and overlays "
                "the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). The cell "
                "asserts identical image embeddings on eight test photographs (VER4) — a contract check, so it stays a hard "
                "check: a reloaded artifact that differs would be a broken export.\n\n"
                "**Predict before running:** the reloaded pipeline is rebuilt from files on disk — a fresh base plus the adapter "
                "tensors. Will its embeddings equal the in-memory adapted model's exactly, or only approximately?"
            ),
            "code": (
                "import shutil\n\n"
                "adapted_classifications = [pipe.zero_shot_classify(image_path, candidate_labels) for image_path in shape_images]\n"
                "adapted_retrievals = [pipe.retrieve(query, shape_images, top_k=len(shape_images)) for query in shape_labels]\n"
                "adapted_scene = evaluation_report({{'classifications': adapted_classifications, 'retrievals': adapted_retrievals, 'gallery_ids': [p.name for p in shape_images]}}, targets, sample_kind='synthetic')\n"
                "for image_path, before, after in zip(shape_images, classifications, adapted_classifications, strict=True):\n"
                "    print({{'image': image_path.name, 'frozen': [(s.label, round(s.score, 3)) for s in before[:2]], 'adapted': [(s.label, round(s.score, 3)) for s in after[:2]]}})\n"
                "print({{'scene_after_adaptation': {{m['id']: round(m['value'], 3) for m in adapted_scene['metrics']}}, 'verdict': adapted_scene['verdict']}})\n"
                "with open('outputs/{stem}_shapes.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump({{'frozen': classification_rows, 'adapted': [{{'image': p.name, 'scores': [asdict(x) for x in ranking]}} for p, ranking in zip(shape_images, adapted_classifications, strict=True)]}}, handle, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = SiglipPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = pipe.embed_image([r['image'] for r in test_records[:8]])\n"
                "after = reloaded.embed_image([r['image'] for r in test_records[:8]])\n"
                "parity = {{'max_abs_difference': float(np.abs(before - after).max()), 'identical_rows': int((np.abs(before - after).max(axis=1) < 1e-5).sum()), 'of': int(before.shape[0])}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_rows'] == parity['of']\n\n"
                "write_provenance('outputs/provenance.json', pipeline=pipe)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'fetched_this_run': fetched, 'weight_file': MODEL_FILENAME, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': MODEL_SHA256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'base_url': CORPUS_BASE_URL, 'bytes': CORPUS_BYTES, 'pinned_photographs': len(SAMPLE_RECORDS), 'species': {{k: list(v) for k, v in SPECIES.items()}}}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'shapes': {{'names': [p.name for p in shape_images], 'sha256': shape_sha256, 'labels': shape_labels, 'candidate_labels': candidate_labels}}, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene}},\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': numpy.__version__, 'device': str(pipe.device), 'dtype': 'float32', 'checkpoint_source': pipe.checkpoint_source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "**What to notice:** the adapter's size and tensor count, `identical_rows` against `of`, and the list of files in "
                "`outputs/`.\n\n"
                "<details><summary>Check your reasoning</summary>Exactly, to the tolerance printed: the recorded run reported "
                "identical embeddings on all eight test photographs. The adapter holds the trained tensors byte for byte "
                "(safetensors, no pickle), the base is the same verified snapshot, and the overlay replaces exactly those "
                "tensors, so the arithmetic is the same. On a GPU a different kernel choice could in principle move the last "
                "digit, which is why the check uses a tolerance of 1e-5 rather than bit equality.</details>"
            ),
        },
        {
            "md": (
                "## 10. Change one thing: how many blocks to train (optional)\n\n"
                "*Evaluation practice.* A **Predict → Change one thing → Run → Observe → Explain** activity, off by default so "
                "Run all is unaffected. Set `RUN_EXPERIMENT = True`, change **one** field — by default it trains the last **four** "
                "vision blocks instead of two — and run this cell (it needs the variables of Sections 4–9, so run it after them). "
                "The experiment loads its **own** pipeline from the verified snapshot, so it starts from the frozen model and "
                "never touches the default `pipe`; it writes only to `outputs/{stem}_experiment/`, prints the default and the "
                "changed run side by side, and checks that the default exports (adapter, evaluation report, result) are "
                "byte-identical afterwards. It loads a second copy of the model (about 0.8 GB) and trains again, so expect a few "
                "minutes on a T4 and much longer on CPU.\n\n"
                "**Predict:** with four trainable blocks — 35,440,128 parameters instead of 21,264,384, an adapter of about "
                "142 MB instead of 85 MB — will held-out mAP rise above the default run's, stay level, or fall? What should "
                "epoch 0 of the experiment show?"
            ),
            "code": (
                "RUN_EXPERIMENT = False  # @param {{type:\"boolean\"}}\n"
                "EXPERIMENT_TRAINABLE_VISION_LAYERS = 4  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n\n"
                "if not RUN_EXPERIMENT:\n"
                "    print({{'experiment': 'skipped (RUN_EXPERIMENT = False); the default path above is complete'}})\n"
                "else:\n"
                "    canonical_files = {{'adapter': artifact_dir / 'adapter.safetensors', 'evaluation_report': Path('outputs/{stem}_evaluation_report.json'), 'result': Path('outputs/{stem}_result.json')}}\n"
                "    canonical = {{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in canonical_files.items()}}\n"
                "    experiment_dir = Path('outputs/{stem}_experiment')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_dir.mkdir(parents=True)\n"
                "    # Its own pipeline from the verified snapshot: the experiment starts from the frozen model and the default pipe is untouched.\n"
                "    experiment_pipe = SiglipPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n"
                "    experiment_result = experiment_pipe.adapt(train_records, val_records, epochs=EXPERIMENT_EPOCHS, lr=EXPERIMENT_LEARNING_RATE, batch_size=BATCH_SIZE, trainable_vision_layers=EXPERIMENT_TRAINABLE_VISION_LAYERS, class_names_map=display_names, progress=report)\n"
                "    experiment_test = experiment_pipe.evaluate(test_records, classes=classes, class_names_map=display_names)\n"
                "    experiment_dir_adapter = experiment_pipe.save_artifact(experiment_dir / 'adapter', metadata={{'tutorial': '{stem}', 'experiment': True, 'data_source': data_source}})\n"
                "    side_by_side = {{\n"
                "        'settings': {{'default': {{'trainable_vision_layers': adapt_result['trainable_vision_layers'], 'epochs': adapt_result['epochs'], 'lr': adapt_result['lr']}}, 'experiment': {{'trainable_vision_layers': EXPERIMENT_TRAINABLE_VISION_LAYERS, 'epochs': EXPERIMENT_EPOCHS, 'lr': EXPERIMENT_LEARNING_RATE}}}},\n"
                "        'trainable_parameters': {{'default': adapt_result['n_trainable'], 'experiment': experiment_result['n_trainable']}},\n"
                "        'epoch0_validation_t2i_map (frozen model)': {{'default': round(adapt_result['history'][0]['val']['t2i_map'], 3), 'experiment': round(experiment_result['history'][0]['val']['t2i_map'], 3)}},\n"
                "        'best_epoch': {{'default': adapt_result['best_epoch'], 'experiment': experiment_result['best_epoch']}},\n"
                "        'test': {{metric: {{'frozen': round(frozen_test[metric], 3), 'default': round(adapted_test[metric], 3), 'experiment': round(experiment_test[metric], 3)}} for metric in METRICS}},\n"
                "        'adapter_bytes': {{'default': artifact_manifest['files'][0]['bytes'], 'experiment': (experiment_dir_adapter / 'adapter.safetensors').stat().st_size}},\n"
                "    }}\n"
                "    for key, row in side_by_side.items():\n"
                "        print({{key: row}})\n"
                "    with open(experiment_dir / 'experiment_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'side_by_side': side_by_side, 'history': experiment_result['history'], 'test_metrics': experiment_test}}, handle, indent=2, ensure_ascii=False)\n"
                "    unchanged = {{name: hashlib.sha256(path.read_bytes()).hexdigest() == canonical[name] for name, path in canonical_files.items()}}\n"
                "    if not all(unchanged.values()):\n"
                "        raise RuntimeError(f'the experiment changed a default export: {{unchanged}}')\n"
                "    print({{'default_exports_unchanged': unchanged, 'experiment_outputs': str(experiment_dir)}})\n"
                "    del experiment_pipe"
            ),
        },
        {
            "md": (
                "**Observe → Explain.** Compare the two `test` columns with the frozen one, and read `best_epoch` for both.\n\n"
                "<details><summary>Check your reasoning</summary>Epoch 0 of the experiment must equal epoch 0 of the default run "
                "(validation mAP 0.68 in the recorded run): both are the frozen model, which is exactly what the restart-from-base "
                "rule guarantees. No experiment run is recorded yet, so the held-out result is yours to explain. Twice the "
                "trainable blocks give the model more capacity to fit 216 photographs, which can raise the held-out score or "
                "memorise faster and peak earlier; with 48 validation photographs choosing the epoch and 96 test photographs "
                "scoring it, a difference of one or two photographs (0.01–0.02 accuracy) is within what a different seed could "
                "produce. Say which it was on your run, and whether the extra 57 MB of adapter bought anything.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model is already a usable zero-shot classifier on six bird species it was never told about — far above "
        "the two non-neural baselines — and a bounded fine-tuning of the vision tower's last two blocks and head on 216 "
        "photographs moves the retrieval view clearly (text-to-image mAP 0.72 → 0.89 in the build record) and the top-1 "
        "accuracy by eight photographs (0.76 → 0.84), with the gain concentrated on the species the prompt separated worst, "
        "and an 85 MB adapter that reloads to identical embeddings. That is the claim: the adaptation contract works end to "
        "end on a real labelled photograph set, and the numbers it produces are read on three metrics, per species, on two "
        "prompt sets, against two non-neural baselines and the frozen model rather than in isolation.\n\n"
        "The test split is 96 photographs from one seeded draw of one sample, the validation split that picks the epoch is 48, "
        "the metrics are three reference-based scores (own numpy implementations; none a human judgement), accuracy moves in "
        "steps of one photograph, and the build record's own epoch history shows the estimate's fragility: validation accuracy on the "
        "48 photographs moved between 0.71 and 0.85 from one epoch to the next. So a gain here says the "
        "contract works, not that the adapted model is better on your photographs, that its scores are calibrated, or that a "
        "sigmoid score is a probability of being right — it still returns a best label for every image, and it can be wrong "
        "confidently. Fine-tuning on a narrow set can also erode the model elsewhere; the drawn shapes re-scored in Section 9 "
        "are three images of evidence about that, not a measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** the majority floor, the colour neighbour and the frozen "
        "model's score on *your* labels are the numbers to read before any adapted one, per class and on the retrieval view. "
        "**Leakage:** keep every photograph in one split (the contract de-duplicates by decoded pixels) and split by "
        "photographer or session when your images come from few sources — the sample's observer overlap is printed for exactly "
        "that reason. **Prompts:** the classifier is the prompt as much as the tower; a second prompt set is scored here so the "
        "difference is visible, and a label an annotator wrote is not a prompt the model understands.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled photograph set, validate "
        "the demonstrated dataset contract without leakage, execute the inference contract and a bounded fine-tuning, evaluate "
        "against two trivial baselines and the frozen model on an image-disjoint split, and emit the shown machine-readable "
        "artifacts — without the repository being reachable. It does **not** establish benchmark superiority, zero-shot "
        "accuracy on any other population or camera, calibration, or production fitness.\n\n"
        "**Optional experiments (off by default; each names its field and what to run):** Section 10 trains four blocks "
        "instead of two in its own pipeline and prints it beside the default run — change `EXPERIMENT_TRAINABLE_VISION_LAYERS`, "
        "`EXPERIMENT_EPOCHS` or `EXPERIMENT_LEARNING_RATE` there (one at a time) and run that cell again; every run starts "
        "from the frozen model and writes only to `outputs/{stem}_experiment/`. Changing `EPOCHS`, `LEARNING_RATE` or "
        "`TRAINABLE_VISION_LAYERS` in Section 7 and choosing **Run after** also starts from the pinned base, but it replaces "
        "the default results and exports. BYOD: `USE_BYOD` and `BYOD_PATH` in Section 4, then **Run after** from Section 4, "
        "and read the two baselines before the adapted number.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"This notebook needs a Linux x86_64 runtime\"** — you are on Windows, macOS or an ARM "
        "machine. Use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n"
        "- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a "
        "complete environment is reused, an incomplete one is finished. If it repeats, the network is blocking or altering "
        "`files.pythonhosted.org` or `pypi.org`.\n"
        "- **\"The isolated environment's Python process exited\"** — usually out of memory (Section 9 holds the adapted and "
        "the reloaded model, Section 10 a third copy). Restart the session and choose **Run all**; prefer a GPU runtime, and "
        "leave Section 10 off on a small CPU runtime.\n"
        "- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the "
        "cells after it keep working. After a session restart, run from the top.\n"
        "- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — `verify_snapshot` names the file. "
        "Delete the snapshot folder Section 3 prints as `weights_dir` and run Section 3 again; the snapshot comes from "
        "`huggingface.co`.\n"
        "- **Section 4 cannot fetch a photograph, or one fails its digest** — `fetch_corpus` names the photo; the open-data "
        "bucket may be briefly unreachable. Run Section 4 again (cached photographs are re-hashed, not re-downloaded); delete "
        "`weights/inat-birds/` if a cached file is corrupt.\n"
        "- **CUDA out of memory in Section 7 or 10** — set `BATCH_SIZE = 8` in Section 7 and choose **Run after** from "
        "Section 7 (the numbers will differ a little from the recorded run), or use a runtime with more GPU memory.\n"
        "- **BYOD: \"BYOD_PATH … does not exist\"** — the path is relative to the working directory printed in the message; "
        "give the zip or the folder holding `labels.csv`.\n"
        "- **BYOD: \"the upload dialog exists only in Google Colab\"** — on Kaggle or Jupyter, put the zip in the runtime "
        "(or attach it as a dataset) and set `BYOD_PATH`.\n"
        "- **BYOD: \"Upload exactly one .zip file\"** — the dialog was cancelled or several files were chosen; run Section 4 "
        "again.\n"
        "- **BYOD: \"labels.csv line N (file …): …\"** — that row names a file that is not in the zip or folder, an id or "
        "label outside the allowed characters, or an image Pillow cannot decode; fix that row.\n"
        "- **BYOD: \"split leaves … supply at least N photographs\"** — a label is too small for the split; the message "
        "states the minimum for your number of labels (12 for two labels).\n\n"
        "## Glossary\n\n"
        "- **Zero-shot classification** — classifying with no training on the classes: each label is put into a prompt and "
        "the photograph is scored against every prompt.\n"
        "- **Prompt / prompt template** — the sentence a label is put into before the text tower embeds it "
        "(`This is a photo of {{label}}.`); a different wording gives a different classifier.\n"
        "- **Embedding** — the unit-length vector a tower produces for a photograph or a text; two embeddings are compared "
        "by their cosine similarity.\n"
        "- **`logit_scale` / `logit_bias`** — two learned numbers that turn a cosine into the model's logit "
        "(cosine × e^`logit_scale` + `logit_bias`); they stay frozen here.\n"
        "- **Sigmoid score** — the logit squashed into 0..1 for one image–prompt pair on its own; scores for different labels "
        "do not sum to 1 and are not calibrated probabilities.\n"
        "- **Sigmoid loss** — SigLIP's training loss: for every photograph and every class prompt, push the logit up for the "
        "true class (+1) and down for the others (−1).\n"
        "- **Vision tower blocks / attention-pool head** — the image tower is twelve transformer blocks followed by a "
        "post-layernorm and an attention-pooling head that makes one vector per image; Section 7 trains the last two blocks "
        "and the head.\n"
        "- **Accuracy** — the share of photographs whose top label is right.\n"
        "- **Macro F1** — the unweighted mean over classes of the harmonic mean of precision and recall, so a class the "
        "model ignores pulls it down.\n"
        "- **Average precision (AP) / text-to-image mAP** — rank all test photographs by one class prompt's score and "
        "average the precision at each true photograph; mAP averages that over the classes. It reads the ranking, not just "
        "the top label.\n"
        "- **Majority floor / colour nearest neighbour** — two non-neural baselines: always answer the most frequent "
        "training label; answer the label of the training photograph with the closest 3×3 mean-colour grid.\n"
        "- **Epoch / learning rate** — one pass over the training photographs; the size of each update step.\n"
        "- **Validation selection** — choosing the epoch on a split that is neither trained on nor used for the final "
        "score.\n"
        "- **Held-out test split** — photographs never used for training or selection; the numbers to report.\n"
        "- **Leakage** — the same (or near-identical) photograph, or the same source, in training and test; the split "
        "de-duplicates by decoded pixels and reports observer overlap.\n"
        "- **Adapter** — the trained tensors only (safetensors), overlaid on the pinned base at load time.\n"
        "- **Reload parity** — the exported adapter, loaded into a fresh pipeline from disk, gives the same embeddings.\n"
        "- **BYOD** — bring your own data: your labelled photographs through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run, not the recorded one:\n\n"
        "- The task was ___ on ___ test photographs of ___ classes.\n"
        "- The majority floor scored ___, the colour neighbour ___ and the frozen model ___ (accuracy); the frozen verdict "
        "was ___.\n"
        "- After fine-tuning ___ blocks, test mAP moved from ___ to ___ and accuracy from ___ to ___; the adaptation verdict "
        "was ___.\n"
        "- The species that changed most was ___, because ___.\n"
        "- One reason not to trust this number on my own photographs yet: ___ (for example the size of the test split, "
        "observer overlap, or the prompts).\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/siglip-v1-zero-shot-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/siglip-v1-zero-shot-pipeline/blob/main/MODEL_CARD.md\n"
        "- Sample dataset card (synthetic shapes): https://github.com/kurtvalcorza/siglip-v1-zero-shot-pipeline/blob/main/examples/sample-data/DATASET_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/huggingface/transformers\n"
        "- Sigmoid Loss for Language Image Pre-Training (Zhai, Mustafa, Kolesnikov and Beyer, ICCV 2023): https://arxiv.org/abs/2303.15343\n"
        "- The SigLIP 2 successor row in this fleet, sharing the code shape: https://github.com/kurtvalcorza/siglip2-vision-language-pipeline\n"
        "- iNaturalist open data (CC0 photographs, each observer's own licence): https://www.inaturalist.org/pages/developers — bucket https://inaturalist-open-data.s3.amazonaws.com/\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
