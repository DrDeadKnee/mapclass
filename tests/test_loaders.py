"""Loader tests: GCS-only singleton model loader + requires_grad data loader.

Mocked-surface tests (singleton identity, cache-hit) run with only the stdlib.
The full tensor-shape/range test requires the pinned torch+transformers stack
(built by Plan 01-01); it is skipped when those deps are absent and is
exercised by the orchestrator's post-merge pinned suite.
"""

import importlib
import os
import sys
import types
from unittest import mock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _torch_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


# --- model_loader: GCS-only, eager attn, singleton -------------------------
def test_model_loader_source_uses_eager_attn_and_no_naflex():
    import mapclass.model_loader as ml
    import mapclass.data_loader as dl

    ml_src = open(ml.__file__, "r", encoding="utf-8").read()
    dl_src = open(dl.__file__, "r", encoding="utf-8").read()

    # AC: literal attn_implementation="eager" present in model_loader.
    assert 'attn_implementation="eager"' in ml_src
    # AC: NaFlex path forbidden in both loaders.
    for forbidden in ("Siglip2ImageProcessor", "pixel_attention_mask", "spatial_shapes"):
        assert forbidden not in ml_src, f"{forbidden} found in model_loader"
        assert forbidden not in dl_src, f"{forbidden} found in data_loader"
    # AC: data_loader returns the requires_grad_() tensor with no
    # clone/detach/re-.to() applied to the returned tensor path.
    returned_block = dl_src.split("img_tensor = inputs")[1]
    assert ".clone()" not in returned_block
    assert ".detach()" not in returned_block


def test_model_loader_returns_cached_singleton_same_identity():
    import mapclass.model_loader as ml

    importlib.reload(ml)
    ml.reset_singleton()

    fake_model = object()
    fake_proc = object()

    with mock.patch.object(ml, "_verify_gpu", lambda: None), mock.patch.object(
        ml, "_download_model_mirror", lambda d: d
    ):
        # Stub transformers so no real weights are loaded.
        fake_tf = types.ModuleType("transformers")
        fake_tf.AutoModel = mock.Mock()
        fake_tf.AutoModel.from_pretrained.return_value.to.return_value.eval.return_value = (
            fake_model
        )
        fake_tf.AutoProcessor = mock.Mock()
        fake_tf.AutoProcessor.from_pretrained.return_value = fake_proc

        with mock.patch.dict(sys.modules, {"transformers": fake_tf}):
            m1, p1 = ml.get_model_and_processor(local_dir="/tmp/x")
            m2, p2 = ml.get_model_and_processor(local_dir="/tmp/x")

    assert m1 is m2 is fake_model
    assert p1 is p2 is fake_proc
    # Loaded exactly once (singleton).
    assert fake_tf.AutoModel.from_pretrained.call_count == 1
    assert fake_tf.AutoProcessor.from_pretrained.call_count == 1
    ml.reset_singleton()


# --- data_loader: cache hit on 2nd call (no GCS download) ------------------
def test_data_loader_second_call_hits_local_cache_no_gcs():
    import mapclass.data_loader as dl

    importlib.reload(dl)

    fake_blob = mock.Mock()
    download_calls = {"n": 0}

    def _download(path):
        download_calls["n"] += 1
        with open(path, "wb") as fh:
            fh.write(b"\xff\xd8\xffcached-bytes")

    fake_blob.download_to_filename.side_effect = _download
    fake_bucket = mock.Mock()
    fake_bucket.blob.return_value = fake_blob

    class _Pil:
        size = (200, 100)

        def convert(self, _mode):
            return self

    fake_pil_pkg = types.ModuleType("PIL")
    fake_image = types.ModuleType("PIL.Image")
    fake_image.open = lambda *_a, **_k: _Pil()
    fake_pil_pkg.Image = fake_image

    proc = mock.Mock()
    proc.tokenizer.model_max_length = 64
    proc.return_value.to.return_value = {
        "pixel_values": mock.Mock(requires_grad_=lambda: "TENSOR"),
        "input_ids": "IDS",
        "attention_mask": "MASK",
    }

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(dl.config, "LOCAL_IMAGE_CACHE_DIR", tmp), \
             mock.patch.object(dl, "get_bucket", lambda: fake_bucket), \
             mock.patch.dict(
                 sys.modules,
                 {"PIL": fake_pil_pkg, "PIL.Image": fake_image},
             ):
            entry = {"id": "RUMSEY~CACHE"}
            dl.load_slice(entry, "a river", device="cpu", processor=proc)
            dl.load_slice(entry, "a river", device="cpu", processor=proc)

    # First call downloads once; second call is a pure cache hit (0 extra).
    assert download_calls["n"] == 1
    assert fake_bucket.blob.call_count == 1


# --- full tensor contract (requires the pinned stack) ---------------------
def test_load_slice_returns_requires_grad_tensor_in_range():
    if not _torch_available():
        print(
            "SKIP test_load_slice_returns_requires_grad_tensor_in_range: "
            "torch/transformers not installed in this worktree (Plan 01-01 "
            "builds the pinned venv; orchestrator runs this post-merge)."
        )
        return

    import torch  # noqa: F401

    import mapclass.data_loader as dl

    importlib.reload(dl)

    # A real-shaped processor double: emits a [-1,1] (1,3,384,384) tensor.
    class _Proc:
        class tokenizer:
            model_max_length = 64

        def __call__(self, **kwargs):
            import torch as _t

            class _Batch(dict):
                def to(self, _device):
                    return self

            px = (_t.rand(1, 3, 384, 384) * 2) - 1  # in [-1, 1]
            return _Batch(
                pixel_values=px,
                input_ids=_t.zeros(1, 64, dtype=_t.long),
                attention_mask=_t.ones(1, 64, dtype=_t.long),
            )

    class _Pil:
        size = (200, 100)

        def convert(self, _mode):
            return self

    fake_pil_pkg = types.ModuleType("PIL")
    fake_image = types.ModuleType("PIL.Image")
    fake_image.open = lambda *_a, **_k: _Pil()
    fake_pil_pkg.Image = fake_image

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cached = os.path.join(tmp, "RUMSEY~T.jpg")
        with open(cached, "wb") as fh:
            fh.write(b"\xff\xd8\xffx")
        with mock.patch.object(dl.config, "LOCAL_IMAGE_CACHE_DIR", tmp), \
             mock.patch.dict(
                 sys.modules,
                 {"PIL": fake_pil_pkg, "PIL.Image": fake_image},
             ):
            img_t, ids, mask, pil = dl.load_slice(
                {"id": "RUMSEY~T"}, "a river", device="cpu", processor=_Proc()
            )

    assert tuple(img_t.shape) == (1, 3, 384, 384)
    assert img_t.requires_grad is True
    assert float(img_t.min()) >= -1.01
    assert float(img_t.max()) <= 1.01
