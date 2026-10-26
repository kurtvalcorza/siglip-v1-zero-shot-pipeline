"""Tier C review findings T81-1..T81-5 (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "siglip_v1_zero_shot_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _markdown():
    return "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")


def test_t81_1_dat9_and_no_never_told_claim():
    md = _markdown()
    assert "never told about" not in md and "it is a trained zero-shot classifier" not in md
    assert "**Overlap with the training data:**" in md and "inferred, not verified" in md


def test_t81_2_selector_is_validation_accuracy_everywhere():
    md = _markdown()
    assert "the metric the epoch was selected on" not in md
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "why the epoch is selected on mAP" not in record


def test_t81_3_tuning_basis_disclosed():
    assert "does not record which" in _markdown()
    assert "configuration that gained on the held-out split" not in _markdown()


def test_t81_4_recorded_runs_are_named():
    md = _markdown()
    assert "build record" not in md
    assert "`7e07ba5c`, commit `ec6b1bc`, 2026-10-09" in md


def test_t81_5_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    assert "[1] X. Zhai" in _markdown()
    md = _markdown()
    assert "`DIMER_ISOLATED_ENV`" in md and "classes it was never trained on" not in md
