"""Regression tests for the 2026-10-05 notebook review findings (SIG-M1..M4, SIG-m1).

Every test needs only CI's dependencies. The notebook's own cell sources are executed with stand-ins where a
model would be needed; the torch-backed test of ``SiglipPipeline.adapt`` skips cleanly when torch is absent.
The carried ``samples`` module is imported under a private alias package so these tests run without torch and
never replace the real ``siglip_v1_pipeline`` package for the rest of the session.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "siglip_v1_zero_shot_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PKG_DIR = ROOT / "src" / "siglip_v1_pipeline"
STEM = "siglip_v1_zero_shot"


def _samples():
    """`samples.py` (and `config.py`) imported torch-free under an alias package."""
    name = "_siglip_review_alias"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(PKG_DIR)]
        sys.modules[name] = pkg
    return importlib.import_module(f"{name}.samples")


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _learner_text(notebook: dict) -> str:
    """Markdown plus every code cell except the carried package modules (whose f-strings are legitimate)."""
    return "\n".join(
        c["source"]
        for c in notebook["cells"]
        if not c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    )


# --- SIG-M1: no in-kernel install, no restart, idempotent Section 1 ------------------------------------------


def test_sig_m1_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source


def test_sig_m1_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    spec = importlib.util.spec_from_file_location("_review_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def _run_bootstrap(source: str, namespace: dict) -> None:
    exec(compile(source, "<section 1>", "exec"), namespace)


def test_sig_m1_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    try:
        (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    except OSError as exc:  # Windows without the symlink privilege (WinError 1314); the cell targets Linux runtimes
        pytest.skip(f"cannot create a symlink here: {exc}")
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    displayed: list = []
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: displayed.append(a)
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    _run_bootstrap(source, namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        _run_bootstrap(source, namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        routed = namespace["_route_to_isolated_runtime"](["x = 1\n"])
        assert routed == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


# --- SIG-M2: every adaptation starts from the pinned base; experiments have their own state ---------------------


def test_sig_m2_adapt_restores_the_base_before_training_starts():
    text = (PKG_DIR / "pipeline.py").read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    def restore_base(")]
    restore = adapt.index("restored = self.restore_base()")
    snapshot = adapt.index("_remember_base(self._base_state, model, names)")
    first_epoch = adapt.index("entry: dict[str, Any] = {")
    training = adapt.index("for epoch in range(1, epochs + 1):")
    assert restore < snapshot < first_epoch < training
    load = text[text.index("    def load_artifact(") : text.index("    def from_artifact(")]
    assert load.index("self.restore_base()") < load.index("self.model.load_state_dict(merged, strict=True)")


def test_sig_m2_experiment_is_off_by_default_with_its_own_pipeline_and_outputs(notebook):
    source = _cell(notebook, "RUN_EXPERIMENT = False")
    assert "experiment_pipe = SiglipPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)" in source
    assert f"Path('outputs/{STEM}_experiment')" in source
    assert "raise RuntimeError(f'the experiment changed a default export: {unchanged}')" in source
    assert not re.search(r"(?<!experiment_)pipe\.adapt\(", source)  # the default pipeline is never trained again here
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    assert "**Predict → Change one thing → Run → Observe → Explain**" in markdown
    assert "optional experiments (they do not affect the default path)" not in markdown.lower()


def test_sig_m2_tiny_model_epoch_zero_is_the_frozen_base_after_an_earlier_run():
    torch = pytest.importorskip("torch")
    from siglip_v1_pipeline import VISION_LAYERS, SiglipPipeline

    class TinyProcessor:
        def __call__(self, *, text=None, images=None, **kwargs):
            batch = {}
            if images is not None:
                rows = [np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255.0 for image in images]
                batch["pixel_values"] = torch.tensor(np.stack([np.append(r, 1.0) for r in rows]), dtype=torch.float32)
            if text is not None:
                batch["input_ids"] = torch.tensor([sum(map(ord, t)) % 16 for t in text])
            return batch

    class TinySiglip(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.vision_model = torch.nn.Module()
            self.vision_model.encoder = torch.nn.Module()
            self.vision_model.encoder.layers = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(VISION_LAYERS))
            self.vision_model.post_layernorm = torch.nn.LayerNorm(4)
            self.vision_model.head = torch.nn.Linear(4, 4)
            self.text = torch.nn.Embedding(16, 4)
            self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))
            self.logit_bias = torch.nn.Parameter(torch.tensor(0.0))

        def get_image_features(self, pixel_values):
            x = pixel_values
            for layer in self.vision_model.encoder.layers:
                x = torch.tanh(layer(x)) + x
            return self.vision_model.head(self.vision_model.post_layernorm(x))

        def get_text_features(self, input_ids):
            return self.text(input_ids)

    colours = ["red", "green", "blue", "yellow"]

    def records(n, prefix):
        out = []
        for i in range(n):
            image = Image.new("RGB", (16, 16), colours[i % 4])
            image.putpixel((0, 0), (i, 7, 9))
            out.append({"id": f"{prefix}{i}", "image": image, "label": colours[i % 4]})
        return out

    train, val = records(16, "t"), records(8, "v")
    pipe = SiglipPipeline(TinySiglip(), TinyProcessor(), device="cpu")
    base = {k: v.detach().clone() for k, v in pipe.model.state_dict().items()}
    base_embeddings = pipe.embed_image([r["image"] for r in val])
    first = pipe.adapt(train, None, epochs=3, lr=1e-3, batch_size=4, trainable_vision_layers=1)
    assert first["started_from"] == "pinned base"
    assert any(not torch.equal(base[k], v) for k, v in pipe.model.state_dict().items()), "training changed nothing"
    seen = {}

    def at_epoch_zero(entry):
        if entry["epoch"] == 0:
            seen["embeddings"] = pipe.embed_image([r["image"] for r in val])
            seen["note"] = entry["note"]

    second = pipe.adapt(train, val, epochs=1, lr=1e-3, batch_size=4, trainable_vision_layers=2, progress=at_epoch_zero)
    assert seen["note"] == "frozen model"
    assert np.array_equal(seen["embeddings"], base_embeddings), "epoch 0 must be the frozen base, not the earlier run"
    assert second["started_from"].startswith("pinned base (restored")
    pipe.restore_base()
    assert all(torch.equal(base[k], v) for k, v in pipe.model.state_dict().items())
    # restore_base reports only tensors that differ from the base: the model is back at the base, so a further
    # restore (and an adapt that starts here) claims nothing (t5-base 93a578f: a fresh run said "restored 52 tensors").
    assert pipe.restore_base() == []
    third = pipe.adapt(train, None, epochs=1, lr=1e-3, batch_size=4, trainable_vision_layers=1)
    assert third["started_from"] == "pinned base"


# --- SIG-M3: quality outcomes are reported verdicts; export always runs -------------------------------------------


def test_sig_m3_no_quality_assert_remains(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    asserts = re.findall(r"(?m)^\s*assert .*$", code)
    assert asserts == ["assert parity['identical_rows'] == parity['of']"], asserts  # contract integrity only


def _metrics(acc, f1, mp):
    return {"accuracy": acc, "macro_f1": f1, "t2i_map": mp, "n": 4, "verdict": "measured", "prompt_template": "t", "definitions": {}, "per_class": {"a": {"n": 2, "recall": acc, "ap": mp}, "b": {"n": 2, "recall": acc, "ap": mp}}}


def test_sig_m3_negative_results_are_recorded_and_do_not_stop_the_notebook(notebook, tmp_path, monkeypatch, capsys):
    """Sections 6 and 8 executed with stand-ins: a frozen model below the baselines and an adapted model worse than
    the frozen one complete, print their verdicts and write the evaluation report (stand-in evidence, no model)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()

    class StandInPipe:
        def __init__(self):
            self.calls = 0

        def evaluate(self, records, **kwargs):
            self.calls += 1
            return _metrics(0.2, 0.2, 0.3) if self.calls <= 2 else _metrics(0.1, 0.1, 0.25)

    ns = {
        "pipe": StandInPipe(),
        "train_records": [], "val_records": [], "test_records": [], "classes": ["a", "b"],
        "display_names": {}, "USE_BYOD": True, "SPECIES": {},
        "majority_baseline": lambda *a: {**_metrics(0.5, 0.33, 0.5), "baseline": "majority"},
        "colour_neighbour_baseline": lambda *a: {**_metrics(0.75, 0.7, 0.8), "baseline": "colour"},
        "time": __import__("time"), "json": json,
        "MODEL_ID": "stand-in", "MODEL_REVISION": "0" * 40, "DEFAULT_MODEL_KEY": "stand-in",
        "data_source": "stand-in", "dataset_manifests": {"test": {"digest": "d"}}, "disjoint": {},
        "adapt_result": {"history": [], "n_trainable": 1}, "adapt_seconds": 0.0,
    }
    exec(_cell(notebook, "baseline_majority = majority_baseline(").replace("{stem}", STEM), ns)
    assert ns["frozen_verdict"].startswith("not above both baselines")
    exec(_cell(notebook, "adapted_test = pipe.evaluate(").replace("{stem}", STEM), ns)
    assert ns["adaptation_verdict"] == "worse"
    report = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdicts"] == {"frozen_vs_baselines": ns["frozen_verdict"], "adapted_vs_frozen_t2i_map": "worse"}
    assert "'adaptation_verdict': 'worse'" in capsys.readouterr().out


# --- SIG-M4: the guided layer and infrastructure labelling ----------------------------------------------------


def test_sig_m4_guided_layer_is_present(notebook):
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for heading in (
        "**Who this notebook is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        "## 10. Change one thing",
        "**Learner:**",
    ):
        assert heading in markdown, heading
    assert markdown.count("**Predict") >= 7
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 7
    assert markdown.count("**What to notice:**") >= 6


def test_sig_m4_infrastructure_cells_are_labelled_and_collapsed(notebook):
    infra = [c for c in _code_cells(notebook) if c["metadata"].get("cellView") == "form"]
    modules = [c for c in infra if c["metadata"].get("dimer", {}).get("embedded_module")]
    assert len(modules) == 6
    titled = [c["source"].splitlines()[0] for c in infra if not c["metadata"].get("dimer")]
    assert len(titled) == 3 and all(t.startswith("# @title Infrastructure:") for t in titled), titled
    learner = [c for c in _code_cells(notebook) if c["metadata"].get("cellView") != "form"]
    assert all("# @title Infrastructure" not in c["source"] for c in learner)


def test_sig_m4_no_template_placeholders_leak(notebook):
    text = _learner_text(notebook)
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in text, leftover
    assert "}}" not in markdown


# --- SIG-m1: BYOD contract ----------------------------------------------------------------------------------


def _jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(buffer, "JPEG")
    return buffer.getvalue()


def _zip(path: Path, n: int, *, drop: str | None = None) -> Path:
    rows = ["id,file,label"] + [f"r{i},img{i}.jpg,{'cat' if i % 2 else 'dog'}" for i in range(n)]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("labels.csv", "\n".join(rows) + "\n")
        for i in range(n):
            if f"img{i}.jpg" != drop:
                archive.writestr(f"img{i}.jpg", _jpeg(i))
    return path


def test_sig_m1_stated_minimum_is_what_split_dataset_accepts(tmp_path):
    samples = _samples()
    assert samples.min_byod_photographs(2)["total"] == 12
    assert samples.min_byod_photographs(3)["total"] == 15
    assert samples.min_byod_photographs(4)["per_label"] == 4
    split = samples.split_dataset(samples.load_byod_dataset(_zip(tmp_path / "ok.zip", 12)), seed=42)
    assert {k: len(v) for k, v in split.items()} == {"test": 2, "validation": 2, "train": 8}
    with pytest.raises(ValueError, match="supply at least 12 distinct photographs, 6 per label"):
        samples.split_dataset(samples.load_byod_dataset(_zip(tmp_path / "small.zip", 11)), seed=42)


def test_sig_m1_refusals_name_the_row_file_and_rule(tmp_path):
    samples = _samples()
    with pytest.raises(ValueError, match=r"labels.csv line 5 \(file 'img3.jpg'\): that image file is not in the dataset"):
        samples.load_byod_dataset(_zip(tmp_path / "missing.zip", 12, drop="img3.jpg"))
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("labels.csv", "id,file,label\n")
    with pytest.raises(ValueError, match="no data rows"):
        samples.load_byod_dataset(empty)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "labels.csv").write_text("id,file,label\nr0,a.jpg,dog/cat\n", encoding="utf-8")
    (bad / "a.jpg").write_bytes(_jpeg(0))
    with pytest.raises(ValueError, match=r"labels.csv line 2 \(file 'a.jpg'\): label 'dog/cat'"):
        samples.load_byod_dataset(bad)


def _section_4(notebook: dict, use_byod: bool, path: str) -> str:
    source = _cell(notebook, "USE_BYOD = False").replace("{stem}", STEM)
    source = source.replace("USE_BYOD = False  # @param", f"USE_BYOD = {use_byod}  # @param", 1)
    return source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param", 1)


def _section_4_namespace() -> dict:
    samples = _samples()
    ns = {k: getattr(samples, k) for k in dir(samples) if not k.startswith("__")}
    ns.update({"os": __import__("os"), "Path": Path, "__name__": "__main__"})
    return ns


def test_sig_m1_byod_path_runs_section_4_outside_colab(notebook, tmp_path, monkeypatch, capsys):
    """The notebook's own Section 4 cell on a 12-photograph zip given by BYOD_PATH (no Colab, no upload)."""
    monkeypatch.chdir(tmp_path)
    _zip(tmp_path / "mine.zip", 12)
    ns = _section_4_namespace()
    exec(_section_4(notebook, True, "mine.zip"), ns)
    out = capsys.readouterr().out
    assert ns["raw_rows"] == {"byod": 12, "duplicate_images_dropped": 0, "effective_minimum": ns["min_byod_photographs"](2)}
    assert ns["disjoint"] == {"test": 2, "validation": 2, "train": 8}
    assert "fewer than five held-out test photographs" in out
    assert (tmp_path / "outputs" / f"{STEM}_train.csv").is_file()


def test_sig_m1_byod_without_path_outside_colab_and_bad_path_are_explained(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)  # no google.colab here, as on Kaggle or Jupyter
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(_section_4(notebook, True, ""), _section_4_namespace())
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nowhere.zip' does not exist"):
        exec(_section_4(notebook, True, "nowhere.zip"), _section_4_namespace())


def test_sig_m1_cancelled_or_multiple_uploads_are_refused(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for uploaded, message in (({}, "received 0"), ({"a.zip": b"", "b.zip": b""}, "received 2")):
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        files = types.ModuleType("google.colab.files")
        files.upload = lambda uploaded=uploaded: uploaded
        colab.files = files
        google.colab = colab
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.colab", colab)
        monkeypatch.setitem(sys.modules, "google.colab.files", files)
        with pytest.raises(ValueError, match=message):
            exec(_section_4(notebook, True, ""), _section_4_namespace())


def test_sig_m1_locks_carry_the_slow_tokenizer_backends():
    """The pinned checkpoint names `SiglipTokenizer` (SentencePiece, no fast class in transformers 4.57), which needs
    sentencepiece (`@requires`) and protobuf (`requires_backends` in `__init__`). The isolated environment sees only
    the lock, so both must be in it; main's integration job failed on the missing sentencepiece since fc8b638."""
    config = json.loads((ROOT / "weights" / "siglip-base-patch16-256" / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert config["tokenizer_class"] == "SiglipTokenizer"
    for lock in (LOCK, ROOT / "requirements.lock.txt"):
        pinned = {line.split("==")[0].lower() for line in lock.read_text(encoding="utf-8").splitlines() if "==" in line and not line.startswith(("#", " "))}
        assert {"sentencepiece", "protobuf"} <= pinned, lock.name
