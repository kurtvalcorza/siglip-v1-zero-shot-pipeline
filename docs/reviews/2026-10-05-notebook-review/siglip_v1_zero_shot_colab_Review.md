# SigLIP v1 Zero-Shot E2E Notebook — Review

**Verdict: Needs revision** (four Majors, one Minor)  
**Review date:** 5 October 2026  
**Repository:** `kurtvalcorza/siglip-v1-zero-shot-pipeline`  
**Notebook:** `tutorials/siglip_v1_zero_shot_colab.ipynb`  
**Reviewed commit:** `3641985` (`main`)  
**Notebook Git blob:** `bd4b3cba00e5e0c468a46ea3fd39ba5512b8423b`, generated from `a025d0c`. This is the blob executed in the recorded Kaggle T4 run of 2026-09-20 (commit `cb34e57`).  
**Finding prefix:** `SIG`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2, `ml-worker` `origin/main` at `b9fdd1f`.

> `STATUS.md` declares this notebook **Release-grade**. This review does not change any status; it records why that status does not hold under NOTEBOOK_SPEC 2.2 (SIG-M1) and leaves the decision to the maintainer.

## Executive assessment

The data and evaluation design is the strongest seen in this queue.

- **Data.** 360 CC0 iNaturalist photographs of six bird species, each pinned by byte size and SHA-256 and fetched live. They are split per species 216 / 48 / 96 after pixel-digest de-duplication, and observer overlap is reported.
- **Baselines.** Two non-neural baselines: the majority floor and a 27-number colour nearest neighbour.
- **Frozen model.** Scored on three metrics (accuracy, macro F1, text-to-image mAP), per species, with two prompt sets.
- **Fine-tuning.** A bounded fine-tune of the last two vision blocks and the head, using SigLIP's sigmoid loss and choosing the epoch on validation mAP.
- **Held-out results.** On an image-disjoint test split: accuracy 0.760 → 0.844, mAP 0.718 → 0.891. The gains concentrate on the species the prompts separated worst.
- **Adapter.** Exported with a closed manifest, and reloaded with identical embeddings.
- **Interpretation.** Careful throughout: sigmoid scores are uncalibrated, prompts matter, and 96 photographs give no dispersion estimate.

Four problems stand in the way of `Ready for intended use`, and of the Release-grade status already declared:

1. **`Run all` needs a manual restart (SIG-M1).** The recorded Kaggle run passed "14/14 ok (1 restart after install cell)". `STATUS.md` nonetheless declares the notebook Release-grade.
2. **The suggested experiments silently continue training (SIG-M2).**
   - `pipe.adapt` trains `self.model` in place and starts from whatever weights it currently holds.
   - Re-running Section 7 after changing `TRAINABLE_VISION_LAYERS`, `EPOCHS` or `LEARNING_RATE`, as the "Optional experiments" invite, therefore continues from the adapted weights.
   - Its epoch-0 row is still labelled `"frozen model"`.
   - The result is compared against the Section 6 frozen numbers.
3. **Hard assertions on model quality (SIG-M3).** Two quality checks are written as plain `assert` statements: the frozen model must beat both baselines, and adapted mAP must exceed frozen mAP. A BYOD run whose fine-tune does not help therefore stops with a bare `AssertionError`, before the adapter is exported or reloaded and before `result.json` is written.
4. **Declared `GUIDED`, but the guided layer is absent (SIG-M4).** There is no audience statement, how-to-use, roadmap, glossary, prediction, checkpoint, troubleshooting or conclusion template, and no cell is labelled or collapsed as infrastructure.

One Minor covers the BYOD contract (SIG-m1):

- The stated minimum of 8 photographs is wrong: the effective minimum for two balanced labels is 12.
- A missing file raises a bare `KeyError`.
- The upload works only in Colab.

| Measure (Kaggle T4, blob `bd4b3cba`, 2026-09-20) | Value |
|---|---|
| Code cells | 14/14 after 1 restart; 455.3 s; 855 MB staged |
| Split | 216 / 48 / 96, disjoint; 53 of 222 observers in more than one split (reported) |
| Test accuracy / macro F1 | majority 0.167 / 0.048 · colour neighbour 0.26 / 0.261 · frozen 0.76 / 0.761 · adapted 0.844 / 0.847 |
| Test text-to-image mAP | frozen 0.718 → adapted 0.891 (best epoch 4 of 6, selected on validation mAP) |
| Reload | identical embeddings on 8 test photographs |

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` |
| Declared spec | DIMER Notebook Specification **2.0** |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites: "basic Python and PIL; what a sigmoid score and a cosine similarity are; what accuracy, macro F1 and average precision measure" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU float32, CUDA when present |
| Promised outcomes | <ul><li>pinned install</li><li>verified snapshot (no pickle)</li><li>360 pinned photographs, validated and split without leakage</li><li>inference contract on drawn shapes</li><li>frozen model against two baselines, three metrics, two prompt sets</li><li>bounded fine-tune with validation-mAP epoch selection</li><li>held-out evaluation, per species</li><li>shapes re-scored</li><li>adapter export and reload parity</li><li>provenance</li><li>BYOD zip through "the same validation, seeded stratified split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells"</li><li>"Optional experiments" that "do not affect the default path"</li></ul> |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py` |
| Release status | **Release-grade** (`STATUS.md`, `docs/release-verification.md`) |

### Evidence actually obtained

- **Source inspection:**
  - all 31 cells (14 code; cells 5–15 carry 6 modules);
  - `pipeline.py` (`adapt`: the state at entry, epoch-0 labelling, best-state selection);
  - `samples.py` (`fetch_corpus`, `build_sample_dataset`, `split_dataset`, `load_byod_dataset`);
  - `metrics.py` (baselines);
  - `STATUS.md` and `docs/release-verification.md`.
- **Documented execution evidence:**
  - the Kaggle T4 row and the build-workstation CPU replay row in `docs/release-verification.md`;
  - no executed notebook is committed;
  - no Colab run, no BYOD run and no optional-experiment run are recorded.
- **Direct execution (this review), torch-free** (Python 3.12, `numpy` 2.5.3, `pillow` 11.3.0). The package `__init__` imports torch, so the probe registered an empty package object and imported `config`, `samples` and `metrics` directly.
  - **P3:**
    - fetched all 360 pinned photographs from `inaturalist-open-data.s3.amazonaws.com` with `fetch_corpus` (39,223,447 bytes, every file digest-checked);
    - built the seed-42 split;
    - checked disjointness and observer overlap;
    - recomputed the majority and colour-neighbour baselines with the carried metrics.
  - **P4:** `load_byod_dataset`, `validate_dataset` and `split_dataset` on 7 constructed zips; the upload import outside Colab.
  - **P4b:** the smallest balanced two-label BYOD set that is accepted.
  - Scripts and results are in `siglip_v1_zero_shot_colab_Review_Probes.zip`.
- **Not executed here:** the model. The Hub is unreachable from this container and the locked torch is the multi-GB CUDA build. Model numbers come from the Kaggle record.
- **Learner observation:** none.

## 2. Separate judgments

- **Technical correctness.**
  - Data provenance, leakage control, baselines, epoch selection on validation, the held-out test and the reload checks are correct.
  - P3 reproduced the split, all three split digests' inputs, and both baselines exactly (majority 0.167 / 0.048; colour neighbour 0.26 / 0.261).
  - Defects:
    - the install pattern forces a restart (SIG-M1);
    - in-place adaptation makes re-runs cumulative and mislabels the epoch-0 row (SIG-M2);
    - quality claims are enforced with `assert` (SIG-M3).
- **Promise fulfilment.**
  - The default path delivers everything it promises.
  - The "Optional experiments" promise ("they do not affect the default path") is false in effect (SIG-M2).
  - The BYOD promise of "the same … cells" through to export fails on any non-improving result (SIG-M3), and its stated minimum is wrong (SIG-m1).
- **Learner experience.**
  - Excellent explanatory prose: sigmoid semantics, prompt dependence, why the epoch is chosen on mAP, and the reading order for metrics.
  - No active-learning layer (SIG-M4).
- **Spec conformance.**
  - Unresolved MUSTs:
    - RUN1, RUN10, ENV6, REL2 (SIG-M1), and therefore the Release-grade claim;
    - SRC2 (SIG-M2);
    - RUN8 and DAT14 for BYOD (SIG-M3);
    - DAT12 and DAT19 (SIG-m1).
  - SHOULD deviations: GDL1–GDL14 and UX8 (SIG-M4); GDL10 and UX5 (SIG-M2).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass Run all | cell 3: in-kernel pip plus stale-module guard | Kaggle: 14/14 "(1 restart after install cell)" | Section 1 prose presents the stop as designed | **Not met** (SIG-M1) |
| Verified snapshot, no pickle | cell 17 | 8 files verified | clear | Met |
| 360 pinned photographs, split without leakage | cell 19 | 216/48/96, disjoint, 53/222 observers shared (P3 identical) | observer overlap explained | Met |
| Inference contract on shapes | cell 21 | sanity checks, input manifest, URL refusal | `sample-sanity` stated | Met |
| Frozen vs two baselines, three metrics, two prompt sets | cell 23 | 0.76 vs 0.26 vs 0.167; scientific-name prompts 0.54 | build-record figures cited | Met (enforced by `assert`: SIG-M3) |
| Bounded fine-tune, validation-mAP selection | cell 25 | best epoch 4; 21,264,384 trainable | loss is not evidence; the selector's role is explained | Met (re-run: SIG-M2) |
| Held-out evaluation, per species | cell 27 | 0.718 → 0.891 mAP; 0.760 → 0.844 accuracy | reading order explained | Met (`assert`: SIG-M3) |
| Shapes re-scored; export; reload parity | cell 29 | identical embeddings | "a finding, not a failure" | Met |
| "Optional experiments … do not affect the default path" | cell 30 → re-run cell 25 | continues from adapted weights; epoch 0 labelled "frozen model" | none | **Not met** (SIG-M2) |
| BYOD through the same cells to export | cell 19 | Colab-only; P4b: needs at least 12 photographs, not 8; any non-improving run asserts out before export | contract partly wrong | **Partly met** (SIG-M3, SIG-m1) |
| Guided learning layer | — | absent | — | **Not met** (SIG-M4) |

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 31 cells | The prose is accurate and rich, but there are no predictions, checkpoints, glossary or roadmap (SIG-M4). The learner reads build-record numbers in the prose before seeing their own run. They match the Kaggle run, but a CPU run may differ. |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (P3: data and baselines) | 14/14 after one restart (SIG-M1). P3 reproduced the split and both baselines exactly from a fresh fetch. |
| **Active learning** | Source | See the walkthrough below the table. |
| **Reuse and recovery** | Direct (P4, P4b) + source | See the BYOD results below the table. |

**Active learning (traced in source).** The learner follows the "Optional experiments": set `TRAINABLE_VISION_LAYERS = 4` (or `EPOCHS = 10`, or `LEARNING_RATE = 1e-5`) and re-run cell 25.

1. `adapt` clones `initial_state` from the **current** `self.model` (`pipeline.py`, around line 477). After the default run, that is the adapted model.
2. The epoch-0 row prints `"note": "frozen model"`, but its validation metrics come from the adapted model.
3. Training continues from there: two blocks were already adapted, plus two newly unfrozen ones.
4. Cell 27 then compares the result with the Section 6 frozen scores.

**BYOD results (P4, P4b).**

- **The stated minimum is wrong.** A 10-photograph, two-label zip that meets the stated contract ("at least eight photographs over at least two labels") is refused: `split leaves 6 training records; at least 8 are required`. The smallest balanced two-label set accepted is **12** photographs, which split 8 / 2 / 2.
- **Unguided failures:**
  - a missing image file raises a bare `KeyError: 'img3.jpg'`;
  - a missing `labels.csv`, a single label, and 7 records are each refused with the rule named.
- **The upload is Colab-only.** It fails outside Colab with `ModuleNotFoundError: No module named 'google'`, and the cell has no path field.
- **Negative results crash.** Any BYOD run whose adapted mAP does not exceed the frozen mAP stops at cell 27's `assert` before export (SIG-M3).

## 5. Findings

### Major

#### SIG-M1 — `Run all` needs a manual restart after the install cell, and the notebook is declared Release-grade on that run

- **Cell/section:**
  - cell 3 and the Section 1 prose;
  - the generator's install block;
  - `STATUS.md` and `docs/release-verification.md` ("**Release-grade** … 14/14 ok (1 restart after install cell)").
- **Observed issue:**
  - The cell runs `pip install` for eight pins (`torch` 2.14.0, `transformers` 4.57.6, …) into the running kernel.
  - It stops with `Restart the runtime, then rerun from the top.` when a loaded distribution changed. On the Kaggle image (`torch` 2.10.0, `transformers` 5.0.0 preloaded), it did.
  - The release record treats the restarted run as the REL1/REL10 evidence and promotes the notebook to Release-grade.
- **Consequence:**
  - Learners selecting **Run all** on a stock Colab or Kaggle image meet an error in the first code cell. RUN1, RUN10 and ENV6 forbid this.
  - A Release-grade label on a notebook that cannot run in one pass misstates its readiness to every consumer of the fleet registry.
- **Evidence:** documented in the Kaggle row of `docs/release-verification.md` and in `STATUS.md`; source.
- **Recommended correction:**
  - Adopt the fleet's isolated **uv** environment pattern. Install nothing into the kernel. Use:
    - a pinned `uv` wheel checked by size and SHA-256;
    - `uv venv --managed-python --python 3.12.12`;
    - a hash-locked lock installed with `--require-hashes --only-binary :all:`;
    - stages run in subprocesses.

    Reference implementations: the bioclip2 capstone and the four 2026-10-04/05 standalone notebooks.
  - Re-qualify with a one-pass hosted Run all.
  - The maintainer should decide whether the Release-grade label stands until then. Under NOTEBOOK_SPEC 2.2 the restart-dependent run does not satisfy RUN10 or REL2.
- **Acceptance check:**
  - A fresh Colab or Kaggle runtime completes every code cell in one **Run all** with no restart, recorded with the blob and `restarted: false`.
  - `grep -n "Restart the runtime" tutorials/siglip_v1_zero_shot_colab.ipynb` returns nothing.
  - The status in `STATUS.md` is consistent with that record.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### SIG-M2 — The suggested experiments silently continue training the adapted model, and the epoch-0 row calls it "frozen"

- **Cell/section:**
  - cell 25 (`TRAINABLE_VISION_LAYERS`, `EPOCHS`, `LEARNING_RATE` fields);
  - the Interpretation section's "Optional experiments" (cell 30);
  - `pipeline.py` `SiglipPipeline.adapt`, where `initial_state` is cloned from `model.state_dict()` at entry and epoch 0 is recorded with `"note": "frozen model"`.
- **Observed issue:**
  - `adapt` trains `self.model` in place and starts from the weights it currently holds.
  - The notebook invites the learner to change a field in cell 25 and compare: "set `TRAINABLE_VISION_LAYERS = 4` and compare the artifact size and the held-out mAP; raise `EPOCHS` …; change `LEARNING_RATE` to `1e-5` and read a smaller, steadier gain". It also asserts that these experiments "do not affect the default path".
  - Re-running cell 25 continues from the adapted tower, and labels the starting point "frozen model".
  - Cell 27 then compares the result with the Section 6 frozen scores.
  - The exported adapter's history would describe a single run that never happened.
- **Consequence:**
  - Every suggested comparison is contaminated. "More layers" or "more epochs" appear to help because training simply continued, and "a smaller, steadier gain" at `1e-5` is measured from an already adapted start.
  - The epoch-0 label tells the learner the starting point was frozen when it was not.
- **Evidence:** source (`adapt` reads its initial state from the live model; there is no reset to the verified base). Not executed here: no model access. It is the same pattern as DTR-M3 in the DETR review.
- **Recommended correction:**
  - Make each call to `adapt` start from the verified base: re-load it, or keep the base tensors and restore them at entry.
  - Keep the default run's results in named variables, so that an experiment prints default and changed results side by side.
  - Give the "Optional experiments" an explicit "change one field → re-run cells 25–27" instruction.
- **Acceptance check:**
  - After a default Run all, setting `TRAINABLE_VISION_LAYERS = 4` and re-running cells 25–27 gives an epoch-0 validation mAP equal to the frozen model's (0.68 in the build record).
  - The comparison prints default and changed results side by side.
- **Spec:** SRC2, GDL10, UX5, FT6.

#### SIG-M3 — Model-quality claims are enforced with `assert`, so a non-improving run, BYOD included, aborts before export, reload and provenance

- **Cell/section:**
  - cell 23: `assert frozen_test['t2i_map'] > baseline_majority['t2i_map'] and frozen_test['accuracy'] > baseline_neighbour['accuracy']`;
  - cell 27: `assert adapted_test['t2i_map'] > frozen_test['t2i_map']`.
- **Observed issue:**
  - The notebook turns two empirical outcomes into hard preconditions: the frozen model beats both baselines, and fine-tuning raises held-out mAP.
  - On the pinned sample they hold. On a learner's BYOD data, which goes through "the same … cells", neither is guaranteed. Small sets, near-duplicate classes or an unlucky split can leave the adapted mAP at or below the frozen one.
  - When that happens the run stops with a bare `AssertionError`, before cell 29 exports the adapter, checks reload parity and writes `result.json`.
  - The interpretation text itself warns that "the deltas are sample-sanity evidence … not a benchmark".
- **Consequence:**
  - A BYOD user with a legitimate negative result gets a traceback and no outputs.
  - The notebook teaches that a non-improving fine-tune is an error rather than a finding.
  - An unusual hosted run on the sample would also fail Run all. The tolerance is unstated, and the code makes no claim of GPU determinism.
- **Evidence:** source (the two `assert` statements). Inferred for BYOD; not executed (no model access).
- **Recommended correction:**
  - Replace the two `assert`s with a reported comparison: print the delta and a verdict (`improved`, `no gain` or `worse`), and record it in `evaluation_report.json`.
  - Let export, reload and provenance run regardless.
  - Keep hard checks for contract violations (shapes, finiteness, reload parity), not for empirical outcomes.
- **Acceptance check:**
  - `grep -n "^assert .*t2i_map" tutorials/siglip_v1_zero_shot_colab.ipynb` returns nothing.
  - A run whose adapted mAP is below the frozen mAP completes and writes the adapter, the parity check and `result.json`, with the verdict recorded.
- **Spec:** RUN8, DAT14, EVAL15.

#### SIG-M4 — Declared `GUIDED`, but the guided layer is absent

- **Cell/section:** opening cells 0–1, every section boundary, carried cells 5–15, end of notebook. Generator: `tools/notebook_template.py`.
- **Observed issue:**
  - **Orientation:** no intended-learner statement, no **How to use this notebook**, no roadmap, no Input → Model → Output table.
  - **Glossary:** none, although the notebook relies on sigmoid score, `logit_scale`/`logit_bias`, text-to-image mAP, macro F1, attention-pool head and adapter.
  - **Active learning:** no prediction before the baselines, the fine-tune or the held-out comparison; no checkpoint with a sample answer.
  - **Experiments:** the "Optional experiments" line is not a Predict → Change → Run → Observe → Explain activity, and as shipped it is contaminated (SIG-M2).
  - **Support:** no troubleshooting section (restart, iNaturalist fetch failure, digest mismatch, GPU memory, BYOD errors) and no conclusion template.
  - **Infrastructure:** the six carried modules (cells 5–15) are not labelled **Infrastructure** or collapsed, and no cell has `cellView`.
  - **What is good:** "Look for" notes in Section 4, and outstanding reading guidance in Sections 6–8.
- **Consequence:** a self-paced learner receives expert guidance, but is never asked to commit to an expectation or test their understanding. They also scroll past six carried modules before any learning activity.
- **Evidence:** source inspection; no `cellView` metadata on any cell.
- **Recommended correction:** add the GDL layer in the template, following NOTEBOOK_SPEC 2.2:
  - audience, how-to-use, roadmap, task contract, glossary;
  - predictions before Sections 6, 7 and 8, and collapsed checkpoints;
  - a structured activity built on SIG-M2's restart-from-base fix;
  - troubleshooting and a conclusion scaffold;
  - Infrastructure-titled install and carrier cells with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL14 maps to a named cell, and the install and carrier cells carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL14, UX8.

### Minor

#### SIG-m1 — BYOD contract: the stated minimum is wrong, a missing file raises a bare `KeyError`, and the upload is Colab-only

- **Cell/section:**
  - cell 0 ("at least eight photographs over at least two labels");
  - cell 1 (*Data contract*: "a dataset needs 8..20,000 records");
  - cell 19 (BYOD branch);
  - `samples.py` `load_byod_dataset` and `split_dataset` (`val_fraction` 0.15, `test_fraction` 0.2, `MIN_RECORDS` 8 *training* records after the split).
- **Observed issue:**
  - **Stated minimum too low.** The contract tells the user 8 records suffice. The split then demands 8 *training* records after holding out validation and test, so a two-label set needs at least **12** photographs. At exactly 12, the held-out test is **2** photographs, one per class, and nothing warns about it.
  - **Bare `KeyError`.** A `labels.csv` row naming a file absent from the zip raises `KeyError: 'img3.jpg'`.
  - **Colab-only upload.** The branch imports `google.colab` unconditionally and has no path field. It cannot run on Kaggle or Jupyter, both named as supported, and a cancelled dialog raises a bare `StopIteration`.
- **Consequence:**
  - A user who follows the stated contract is refused with a message about training records they did not know were required.
  - A minimal set produces a two-photograph "held-out" score with no caution.
- **Evidence:** direct (P4, P4b): 8–11 balanced photographs are refused, 12 split 8 / 2 / 2, and a missing file gives `KeyError`. Outside Colab the upload fails with `ModuleNotFoundError: No module named 'google'`.
- **Recommended correction:**
  - State the effective minimum, roughly 12 photographs for two labels and more per extra label, and warn when the test split has fewer than about 5 photographs per class.
  - Wrap the missing-file case in a message naming the `labels.csv` row.
  - Add `BYOD_PATH`, and require exactly one uploaded file.
- **Acceptance check:**
  - The stated minimum matches what `split_dataset` accepts.
  - A missing file produces a message naming its row.
  - A path-based BYOD run completes on Kaggle.
- **Spec:** DAT12, DAT16, DAT19, UX10.

### Suggestions

- **SIG-S1:** Commit the Kaggle executed notebook and `run_summary.json`. The current evidence is table rows only.
- **SIG-S2:** The prose quotes build-record numbers (0.76, epoch 4, 0.72 → 0.89) before the learner's own run. Quote them as "the recorded run" with runtime and date, and let the learner's run speak first.
- **SIG-S3:** 53 of 222 observers contribute to more than one split. The notebook reports this honestly. A note on whether an observer-grouped split changes the result would make the leakage lesson concrete.
- **SIG-S4:** Declare `notebook_spec` 2.2 once the fixes land.

## 6. Readiness

**Needs revision.**

- **Open Majors:** SIG-M1 to SIG-M4.
- **Status:** the current **Release-grade** label rests on a restart-dependent run. Under NOTEBOOK_SPEC 2.2 that does not satisfy RUN10 or REL2. Whether to keep the label is the maintainer's decision; this review only flags it.
- **Gates remaining after the fixes:**
  - a one-pass hosted Run all;
  - an optional-experiment run showing an epoch-0 equal to the frozen model;
  - the REL12 BYOD journey (one compatible and one incompatible zip, path-based and through the dialog), recorded in `docs/release-verification.md`.

## 7. Verified versus inferred

- **Verified by direct execution (torch-free):**
  - all 360 pinned photographs, fetched and digest-checked;
  - the seed-42 split, its disjointness and the observer overlap;
  - both baselines, identical to the Kaggle record;
  - the BYOD minimum-size behaviour, the missing-file `KeyError` and the Colab-only upload.
- **Verified from documented evidence:**
  - the restart on Kaggle;
  - every model number;
  - the Release-grade declaration.
- **Inferred from source:**
  - the in-place re-run contamination and the mislabelled epoch 0 (SIG-M2): `adapt` reads its initial state from the live model;
  - the BYOD abort on a non-improving result (SIG-M3): the two `assert` statements.
- **Most likely to be wrong:** SIG-M3's severity. On the pinned sample the assertions hold and act as regression guards, so a maintainer could keep them for the default path and skip them only for BYOD. The finding stands for BYOD either way.
