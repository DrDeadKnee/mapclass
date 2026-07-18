"""
Offline tests for the Phase-B masked-hex prior: window sampling, blob
masking, model forward/loss, and the windowed dataset. CPU torch only.
"""

import json

import numpy as np
import pytest
import torch

from h3.api import basic_int as h3

from hexprior.dataset import HexWindowDataset
from hexprior.labels import N_COMPOSITE
from hexprior.model import (
    HexPriorModel,
    fourier_features,
    masked_kl_loss,
    masked_top1_accuracy,
)
from hexprior.windows import blob_mask, build_adjacency, local_xy, sample_window


CENTER = h3.latlng_to_cell(40.0, -3.0, 6)
REGION_CELLS = np.array(sorted(h3.grid_disk(CENTER, 5)), dtype=np.uint64)  # 91 cells


@pytest.fixture(scope="module")
def adjacency():
    return build_adjacency(REGION_CELLS)


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------

class TestSampleWindow:
    def test_size_and_uniqueness(self, adjacency):
        rng = np.random.default_rng(0)
        w = sample_window(adjacency, 40, rng)
        assert len(w) == 40 and len(set(w)) == 40

    def test_connected(self, adjacency):
        rng = np.random.default_rng(1)
        w = sample_window(adjacency, 30, rng)
        placed = {w[0]}
        for cell in w[1:]:
            assert any(n in placed for n in adjacency[cell]), "window not connected"
            placed.add(cell)

    def test_clamps_to_component(self, adjacency):
        rng = np.random.default_rng(2)
        w = sample_window(adjacency, 10_000, rng)
        assert len(w) == len(REGION_CELLS)


class TestLocalXY:
    def test_neighbours_at_unit_distance(self, adjacency):
        rng = np.random.default_rng(3)
        w = sample_window(adjacency, 40, rng)
        xy = local_xy(w)
        pos = {c: xy[k] for k, c in enumerate(w)}
        in_window = set(w)
        checked = 0
        for c in w:
            for n in adjacency[c]:
                if n in in_window:
                    d = float(np.linalg.norm(pos[c] - pos[n]))
                    assert d == pytest.approx(1.0, abs=1e-4)
                    checked += 1
        assert checked > 50

    def test_centered(self, adjacency):
        rng = np.random.default_rng(4)
        w = sample_window(adjacency, 30, rng)
        xy = local_xy(w)
        np.testing.assert_allclose(xy.mean(axis=0), [0.0, 0.0], atol=1e-5)


class TestBlobMask:
    @pytest.mark.parametrize("ratio", [0.5, 0.7, 0.95])
    def test_ratio_respected(self, adjacency, ratio):
        rng = np.random.default_rng(5)
        w = sample_window(adjacency, 60, rng)
        m = blob_mask(w, adjacency, ratio, rng)
        assert m.sum() == int(np.clip(round(ratio * 60), 1, 59))
        assert 0 < m.sum() < len(w)

    def test_extreme_ratios_leave_both_classes(self, adjacency):
        rng = np.random.default_rng(6)
        w = sample_window(adjacency, 20, rng)
        for ratio in (0.0, 1.0):
            m = blob_mask(w, adjacency, ratio, rng)
            assert 0 < m.sum() < len(w)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def _tiny_model():
    torch.manual_seed(0)
    return HexPriorModel(d_model=32, n_heads=4, n_layers=2)


def _rand_batch(b=2, k=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    soft = torch.rand(b, k, N_COMPOSITE, generator=g)
    soft = soft / soft.sum(-1, keepdim=True)
    xy = torch.randn(b, k, 2, generator=g) * 4
    mask = torch.zeros(b, k, dtype=torch.bool)
    mask[:, : k // 2] = True
    pad = torch.zeros(b, k, dtype=torch.bool)
    return soft, xy, mask, pad


class TestModel:
    def test_forward_shape_and_finite(self):
        model = _tiny_model()
        soft, xy, mask, pad = _rand_batch()
        logits = model(soft, xy, mask, pad)
        assert logits.shape == (2, 16, N_COMPOSITE)
        assert torch.isfinite(logits).all()

    def test_padding_tokens_do_not_leak(self):
        model = _tiny_model().eval()
        soft, xy, mask, pad = _rand_batch()
        pad = pad.clone()
        pad[:, -4:] = True
        with torch.no_grad():
            a = model(soft, xy, mask, pad)
            soft2 = soft.clone()
            soft2[:, -4:] = 7.0  # garbage at padding positions
            b = model(soft2, xy, mask, pad)
        torch.testing.assert_close(a[:, :-4], b[:, :-4], atol=1e-5, rtol=1e-4)

    def test_fourier_features_shape(self):
        xy = torch.randn(3, 5, 2)
        assert fourier_features(xy, n_freqs=6).shape == (3, 5, 24)


class TestLoss:
    def test_zero_when_prediction_matches_target(self):
        soft, _, mask, _ = _rand_batch()
        logits = torch.log(soft + 1e-12)
        assert masked_kl_loss(logits, soft, mask).item() == pytest.approx(0.0, abs=1e-5)

    def test_matches_manual_kl(self):
        target = torch.tensor([[[0.5, 0.5, 0.0]]])
        logits = torch.log(torch.tensor([[[0.25, 0.25, 0.5]]]))
        mask = torch.ones(1, 1, dtype=torch.bool)
        expected = 0.5 * np.log(0.5 / 0.25) * 2  # Σ t log(t/p); 0·log0 = 0
        assert masked_kl_loss(logits, target, mask).item() == pytest.approx(expected, abs=1e-6)

    def test_only_masked_tokens_scored(self):
        soft, _, _, _ = _rand_batch()
        logits = torch.randn_like(soft)
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[0, 0] = True
        garbage = soft.clone()
        garbage[1] = 1.0 / N_COMPOSITE  # change only unmasked rows
        assert masked_kl_loss(logits, soft, mask).item() == pytest.approx(
            masked_kl_loss(logits, garbage, mask).item(), abs=1e-6)

    def test_top1_accuracy_bounds(self):
        soft, _, mask, _ = _rand_batch()
        acc = masked_top1_accuracy(torch.log(soft + 1e-12), soft, mask)
        assert acc.item() == pytest.approx(1.0)

    def test_one_optimizer_step_reduces_loss(self):
        model = _tiny_model()
        soft, xy, mask, pad = _rand_batch(b=4, k=24)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        first = None
        for _ in range(20):
            loss = masked_kl_loss(model(soft, xy, mask, pad), soft, mask)
            if first is None:
                first = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() < first


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

@pytest.fixture()
def region_npz(tmp_path):
    rng = np.random.default_rng(7)
    counts = rng.integers(0, 200, size=(REGION_CELLS.size, N_COMPOSITE)).astype(np.uint32)
    counts[3] = 0  # one all-nodata hex — must be dropped on load
    path = tmp_path / "toy_res6.npz"
    np.savez_compressed(path, cells=REGION_CELLS, counts=counts,
                        n_pixels=counts.sum(axis=1).astype(np.uint32))
    (tmp_path / "toy_res6.meta.json").write_text(json.dumps({"name": "toy"}))
    return str(path)


class TestHexWindowDataset:
    def test_item_shapes_and_invariants(self, region_npz):
        ds = HexWindowDataset([region_npz], window_size=32, epoch_len=10, seed=1)
        item = ds[0]
        assert item["soft"].shape == (32, N_COMPOSITE)
        assert item["xy"].shape == (32, 2)
        real = ~item["pad"]
        sums = item["soft"][real].sum(-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=0)
        assert not (item["mask"] & item["pad"]).any()
        assert item["mask"].any() and (real & ~item["mask"]).any()

    def test_deterministic_per_index(self, region_npz):
        a = HexWindowDataset([region_npz], window_size=32, epoch_len=10, seed=1)[4]
        b = HexWindowDataset([region_npz], window_size=32, epoch_len=10, seed=1)[4]
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)

    def test_pads_when_region_smaller_than_window(self, region_npz):
        ds = HexWindowDataset([region_npz], window_size=256, epoch_len=2, seed=2)
        item = ds[0]
        # 91 cells minus the dropped all-nodata hex
        assert int((~item["pad"]).sum()) == REGION_CELLS.size - 1
