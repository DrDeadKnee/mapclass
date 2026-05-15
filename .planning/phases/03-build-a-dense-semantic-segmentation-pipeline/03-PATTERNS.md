# Phase 3: Build a dense semantic segmentation pipeline - Pattern Map

**Mapped:** 2026-05-15
**Files analyzed:** 13 (7 new `scripts/seg/` modules + 1 requirements edit + 5 new test files + shared fixtures)
**Analogs found:** 7 with a usable analog / 13 total (6 are spec-driven novel — scaffolding analog only)

> Single-process PyTorch research pipeline. "Role" below uses pipeline-stage
> semantics, not web tiers. Analog quality legend:
> **exact** = same role + data flow; **role-match** = same role, different flow;
> **scaffold** = no behavioural analog, but a structural/idiom analog to copy
> file shape, imports, docstring style, and CLI/test conventions from;
> **none** = genuinely novel, follow RESEARCH.md only.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `scripts/seg/__init__.py` | package-init | n/a | (none needed — empty/`__all__`) | n/a |
| `scripts/seg/backbones.py` | model (backbone) | transform (image→feature grid) | `scripts/finetune_paligemma.py::apply_lora` (PEFT load + Gemma isolation) | role-match |
| `scripts/seg/decoder.py` | model (conv decoder) | transform (feature maps→dense features) | none (UPerNet PPM+FPN is novel) | scaffold |
| `scripts/seg/heads.py` | model (task heads) | transform (features→class logits) | none (thin 1×1 conv heads novel) | scaffold |
| `scripts/seg/model.py` | model (assembled SegModel) | transform (RGB[+prior]→2 logit tensors) | none (wiring is novel) | scaffold |
| `scripts/seg/recursive.py` | orchestrator (c2f inference) | event-driven (pyramid tree walk) | `scripts/tiling.py` (`pyramid.json` schema + parent→child indices, consumed) | role-match (consumer of analog's contract) |
| `scripts/seg/dataset.py` | data (PyramidDataset) | CRUD-read (filesystem→tensors) | `scripts/finetune_paligemma.py::ToonDataset` (torch `Dataset` shape) + `scripts/tiling.py` (`_REQUIRED_FILES`, manifest) | role-match |
| `requirements.txt` | config | n/a | existing `requirements.txt` (append-only edit) | exact |
| `tests/test_seg_backbones.py` | test (unit) | request-response | `tests/test_tiling.py` (offline `tmp_path` synthetic-fixture) | role-match |
| `tests/test_seg_decoder.py` | test (unit) | request-response | `tests/test_tiling.py` | role-match |
| `tests/test_seg_recursive.py` | test (unit) | request-response | `tests/test_tiling.py` (box/quadrant geometry asserts) | exact |
| `tests/test_seg_dataset.py` | test (unit) | request-response | `tests/test_split.py` (`test_split_subtree_preserved` leakage guard) | exact |
| `tests/test_seg_smoke.py` | test (smoke) | request-response | `tests/test_tiling.py::test_pyramid_tile_count` (mini-pyramid build) | role-match |
| `tests/integration/test_seg_online.py` | test (integration) | request-response | `tests/integration/test_satellite_online.py` (gated, `@pytest.mark.integration`) | exact |
| shared seg fixtures | test fixture | n/a | `tests/test_tiling.py::_make_map_dir` + `tests/conftest.py` | exact |

## Pattern Assignments

### `scripts/seg/backbones.py` (model/backbone, transform)

**Analog:** `scripts/finetune_paligemma.py` (`apply_lora`, lines 163-203; load order in `train`, lines 215-235)

**Imports / sys.path pattern** (`finetune_paligemma.py` lines 33-41) — module-level
torch + transformers import; note Phase 3 modules live under `scripts/seg/` and
`pytest.ini` already sets `pythonpath = scripts`, so tests import `from seg.backbones import ...`
with **no** `sys.path` hack. (The `_HERE`/`sys.path.insert` block at lines 43-45 is
for sibling-module imports; not needed inside the `seg` package.)
```python
import torch
from transformers import PaliGemmaForConditionalGeneration
# (peft imported lazily inside the loader, mirroring apply_lora's
#  `from peft import LoraConfig, get_peft_model` at line 175)
```

**Gemma-isolation pattern — THE load-bearing copy (PHASE-03 SC#2)** (`apply_lora`, lines 176-198):
The analog proves the project's established way to (a) load PaliGemma, (b) freeze
Gemma explicitly by name substring, (c) reach the SigLIP-side modules. Lines 178,
191-193 are the exact idiom to mirror for the parameter-graph guarantee:
```python
# scripts/finetune_paligemma.py:178
model.language_model.requires_grad_(False)
# scripts/finetune_paligemma.py:191-193  (name-substring exclusion — mirror this
# as the Gemma-absence ASSERTION in the seg backbone, not just a freeze)
for name, param in peft_model.named_parameters():
    if "language_model" in name:
        param.requires_grad_(False)
```
**Phase-3 divergence (from RESEARCH Pitfall 1/3, Code Examples):** the backbone
must *hold a reference to the vision tower only* and never call the full model.
The analog reaches `peft_model.base_model.model.multi_modal_projector` (line 196) —
the seg path stops one level earlier at `...model.vision_tower` and must NOT touch
the projector (RESEARCH anti-pattern: projector → Gemma token space, wrong dim).
Use the defensive accessor from RESEARCH Code Examples (4.x `.vision_tower` vs
5.x `.model.vision_tower`) — there is **no analog** for this; it is new.

**LoRA target-module contract to preserve** (`apply_lora` docstring, lines 164-188):
`target_modules=["q_proj","k_proj","v_proj","out_proj"]`, `model-id
google/paligemma-3b-pt-224`, save format = PEFT adapter dir written at
`checkpoints/paligemma-terrain/epoch{NN}` (lines 296-298). The backbone loader
must load the *same* base model + `PeftModel.from_pretrained(base, adapter_dir)`
or the Phase-1 fine-tune is silently dropped (RESEARCH Pitfall 2). Adapter dir is
a **CLI/constructor parameter** (Open-Q1 RESOLVED) — mirror the argparse style at
`finetune_paligemma.py` lines 308-325 (`--model-id`/`--adapter-dir`).

**No analog for:** `Backbone` Protocol class, `feature_strides`/`feature_channels`
declared-stride contract, ViT-token→grid reshape, timm DINOv2/Swin construction.
These are spec-driven (RESEARCH Pattern 5 + Code Examples). Flagged below.

---

### `scripts/seg/recursive.py` (orchestrator, event-driven)

**Analog:** `scripts/tiling.py` — this is the *producer* of the contract this
file *consumes*; mirror its manifest vocabulary exactly, do not re-derive geometry (D-04).

**`pyramid.json` schema to read literally** (`tiling.py` `_pyramid_tiles`, lines 84-115;
manifest assembly lines 176-185). Each tile entry is exactly:
```python
# scripts/tiling.py:85-94  — the dict shape recursive.py must index by
{"id": tid, "x": x, "y": y, "size": size, "children": [...],
 "image": f"{tid}_image.png", "land_cover": f"{tid}_land_cover.png",
 "topography": f"{tid}_topography.png"}
# manifest top-level (tiling.py:176-185):
# {"pyramid_id","origin":[ox,oy],"source_size",[...],"scales":[896,448,224],
#  "stride":448,"tiles":[...]}
```
IDs are stable strings: root `"896"`, children `"448_{ci}"` (ci 0..3 in
quadrant order `(0,0),(1,0),(0,1),(1,1)`), grandchildren `"224_{ci}_{gi}"`
(`tiling.py` lines 97-114). The 2×2 quadrant order is load-bearing for the
prior crop — copy it, don't reinvent.

**Quadrant-box crop pattern** (mirror `tests/test_tiling.py::_box`, lines 54-61):
the parent→child pixel offset for the prior crop is `(child.x - parent.x,
child.y - parent.y, +size)` — exactly the box arithmetic the tiler's own tests
use. RESEARCH Code Examples `recursive_predict` is the target structure; its
crop `p[..., cy-oy:cy-oy+s, cx-ox:cx-ox+s]` is the `_box` idiom restated.

**Manifest read idiom** (mirror `tests/test_tiling.py::_load_manifest`, lines 50-51):
`json.loads((pyramid_dir / "pyramid.json").read_text())` — keep this exact form.

**No analog for:** the recursive predict/cache traversal itself, zero cold-start
prior at 896, the 12-channel softmax prior tensor. Spec-driven (D-03/D-03a/D-04,
RESEARCH Pattern 4 + Code Examples).

---

### `scripts/seg/dataset.py` (data/PyramidDataset, CRUD-read)

**Analog (class shape):** `scripts/finetune_paligemma.py::ToonDataset` (lines 67-156).
Copy the `torch.utils.data.Dataset` skeleton: `__init__` builds an index list,
`__len__` returns `len(self.samples)`, `__getitem__` returns a `dict[str, Tensor]`
(lines 141-156). Validate-on-empty pattern (lines 93-94: `raise ValueError` if
no samples) is the project idiom — mirror it for an empty/`test/`-filtered tree.

**Analog (file contract):** `scripts/tiling.py` `_REQUIRED_FILES` (lines 42-44):
```python
# scripts/tiling.py:42-44
_REQUIRED_FILES = ("image.png", "land_cover.png", "topography.png",
                   "sample_weights.json")
```
Per-tile filenames come from the manifest entry keys (`image`/`land_cover`/
`topography`, `tiling.py` lines 91-93). `sample_weights.json` is propagated
byte-identically per pyramid (`tiling.py` lines 187-196) — surface it through
the dataset **unchanged** (CONTEXT §code_context "Established Patterns").

**Split-safety contract:** dataset must take an explicit `train/` root or filter
by `split.json`. The `split.json` schema is `{"test":[ids],...}` keyed `"test"`/
`"train"` (`scripts/build_dataset.py` lines 132-161, `_SPLIT_FILENAME =
"split.json"`). Never `glob("**/pyramid.json")` across the synthetic root
(RESEARCH Pitfall 6). The leakage-assertion analog is in the test section.

**No analog for:** pyramid-aware batching/sampling, prior-channel plumbing.
Discretion under D-06 (RESEARCH §user_constraints).

---

### `scripts/seg/decoder.py`, `heads.py`, `model.py` (model, transform) — NO behavioural analog

These are **spec-driven novel** (RESEARCH "Phase 3's only genuinely new logic is
the wiring"). No FPN/UPerNet/PPM decoder, multi-head, or backbone-decoder
assembly exists in the codebase.

**Scaffold analog to copy (file shape only):** `scripts/finetune_paligemma.py`
top-of-file module docstring style (lines 1-31: purpose, strategy, `Requires:`,
`Usage:`) and `scripts/tiling.py` lines 1-28 (decisions referenced inline as
`D-01`/`D-02`). Mirror this docstring convention — every Phase-1/2 module
opens with a decision-referenced rationale block.

**Source of truth for behaviour:** RESEARCH Pattern 3 (UPerNet PPM+FPN-on-ViT),
Pattern 5 (decoder reads `backbone.feature_strides`/`feature_channels`
dynamically — Pitfall 4), D-01/D-02. Decoder topology, channel widths,
block-tap choice = Claude's Discretion (CONTEXT §Claude's Discretion).
Variant A vs B prior-injection point (D-03a) is novel structural work with no
analog — follow CONTEXT D-03a + RESEARCH Pattern 4 verbatim.

---

### `requirements.txt` (config) — append-only edit

**Analog:** the file itself. Current pins (read 2026-05-15): `transformers>=4.41`,
`peft>=0.10`, `torch` (unpinned), `pytest>=8`, `pytest-mock>=3`; `Pillow`,
`numpy` present. **`timm` is ABSENT** — must be added (RESEARCH Environment
Availability: "Plan Wave 0 must add `timm`"). Follow the existing one-per-line,
`pkg>=ver` (or bare `pkg`) convention. RESEARCH §Standard Stack flags the
transformers 4.x↔5.x path break (Pitfall 3): the plan must decide whether to
re-pin transformers — that is a planning decision, the *edit pattern* (append /
tighten a line) is the only thing the analog dictates.

---

### Test files (`tests/test_seg_*.py`, `tests/integration/test_seg_online.py`)

**Analog (offline unit + synthetic fixture):** `tests/test_tiling.py`.

- **Module docstring** (lines 1-12): state "fully offline … under pytest
  `tmp_path`, no network" + the decision IDs covered. Copy this convention.
- **Synthetic builder** (`_make_map_dir`, lines 40-47): the canonical mini-input
  factory — `Image.new("RGB"/"L", (w,h)).save(...)` + a `sample_weights.json`
  blob. The seg suite needs a **mini-pyramid** builder: build one `_make_map_dir`
  then call `tiling.tile(src)` (real producer) to get a real `pyramid.json` tree
  under `tmp_path` (mirror `test_pyramid_tile_count` lines 85-103). Put this
  shared builder in `tests/conftest.py` (Wave-0 gap) — `conftest.py` already
  hosts `tmp_path`-only fixtures (see `tiny_geotiff` lines 144-172 for the
  write-under-`tmp_path`-return-path idiom).
- **Geometry asserts for `test_seg_recursive.py`** (`_box`/`_contains`/
  `_disjoint`, lines 54-74; `test_nested_alignment` lines 105-135): exact-quadrant
  containment asserts. The "place a known one-hot blob in a parent quadrant,
  assert it lands in the right child" test (RESEARCH Pitfall 5) is a direct
  extension of `test_nested_alignment`'s box arithmetic.

**Analog (split-leakage guard) for `test_seg_dataset.py`:**
`tests/test_split.py::test_no_train_test_intersection` (lines 97-119) and
`tests/test_tiling.py::test_split_subtree_preserved` (lines 203-224):
```python
# tests/test_tiling.py:213-215  — the exact leakage assertion to mirror
assert "test" in test_out.parts
assert "train" not in test_out.parts
assert str(test_out).startswith(str(synthetic / "test"))
```
Assert the dataset never enumerates a path containing a `test/` component.

**Analog (Gemma-absence) for `test_seg_backbones.py`:** no direct test analog;
RESEARCH Code Examples gives the parameter-graph proof — mirror the
name-substring style from `finetune_paligemma.py` lines 191-193:
```python
names = [n for n, _ in seg_model.named_parameters()]
assert not any("language_model" in n for n in names)
assert not any("gemma" in type(m).__name__.lower() for m in seg_model.modules())
```

**Analog (gated integration test):** `tests/integration/test_satellite_online.py`
(read 2026-05-15) — the established pattern for a heavy/artifact-gated test:
module docstring naming the run command, `@pytest.mark.integration` on every
test, `tmp_path` for outputs. For `test_seg_online.py`, skip-with-message when
the Phase-1 adapter / Phase-2 data are absent (Open-Q1 RESOLVED). The
`_SCRIPTS` `sys.path` block (`test_satellite_online.py` lines 19-21) is the
copy-target for integration files (which sit one dir deeper than `pytest.ini`'s
`pythonpath`).

## Shared Patterns

### Provable Gemma exclusion (PHASE-03 SC#2)
**Source:** `scripts/finetune_paligemma.py` lines 178, 191-193 (name-substring
freeze) — extended in Phase 3 to a *named-parameter assertion*.
**Apply to:** `scripts/seg/backbones.py` (`SiglipBackbone`), `scripts/seg/model.py`,
`tests/test_seg_backbones.py`.
```python
# established project idiom (finetune_paligemma.py:191-193):
for name, param in peft_model.named_parameters():
    if "language_model" in name:
        param.requires_grad_(False)
# Phase-3 hardening: also ASSERT no such param/module is reachable from seg_model.
```

### PEFT adapter load order (preserve Phase-1 fine-tune)
**Source:** `scripts/finetune_paligemma.py` lines 215-235 (load `model-id`) +
`apply_lora` 175-188 (LoRA `target_modules`) + save at lines 296-298.
**Apply to:** `scripts/seg/backbones.py`. Load *same* base
`PaliGemmaForConditionalGeneration(model-id="google/paligemma-3b-pt-224")` then
`PeftModel.from_pretrained(base, adapter_dir)`; adapter dir as a parameter
(argparse style: `finetune_paligemma.py` lines 308-325).

### Manifest-driven geometry — never recompute (D-04)
**Source:** `scripts/tiling.py` lines 84-115, 176-185 (the `pyramid.json`
producer); `tests/test_tiling.py` lines 50-74 (`_load_manifest`, `_box`).
**Apply to:** `scripts/seg/recursive.py`, `scripts/seg/dataset.py`,
`tests/test_seg_recursive.py`. Index tiles by `id`, read `x`/`y`/`size`/
`children` from the manifest; derive prior crops from box arithmetic only.

### Offline-first, `tmp_path`-only, decision-referenced tests
**Source:** `tests/test_tiling.py` lines 1-12, 40-47; `tests/conftest.py` lines
1-16, 144-172; `pytest.ini` (`testpaths=tests`, `pythonpath=scripts`,
`integration` marker).
**Apply to:** all `tests/test_seg_*.py`. Offline unit tests in `tests/`,
artifact/heavy ones `@pytest.mark.integration` under `tests/integration/`,
fixtures synthesised under `tmp_path`. Imports are `from seg.x import ...`
(no `sys.path` hack — `pythonpath=scripts` covers it).

### Module docstring convention
**Source:** `scripts/finetune_paligemma.py` lines 1-31, `scripts/tiling.py`
lines 1-28. Purpose + strategy + decision-ID references + `Usage:`/`Requires:`.
**Apply to:** every new `scripts/seg/*.py` and `tests/test_seg_*.py`.

## No Analog Found

Files whose *core behaviour* has no codebase precedent — the planner must drive
these from RESEARCH.md / CONTEXT.md (the scaffold analog above only fixes file
shape, imports, docstring, and test conventions, NOT the algorithm):

| File | Role | Data Flow | Reason / Source of truth |
|------|------|-----------|--------------------------|
| `scripts/seg/decoder.py` | model | transform | UPerNet PPM+FPN-on-ViT is new to the repo. RESEARCH Pattern 3 + Pitfall 4; D-01. |
| `scripts/seg/heads.py` | model | transform | No multi-head conv module exists. D-02; RESEARCH Pattern 3. |
| `scripts/seg/model.py` | model | transform | Backbone+decoder+heads+prior assembly + Variant A/B switch is novel wiring. CONTEXT D-03a/D-06a; RESEARCH Architecture. |
| `scripts/seg/recursive.py` (traversal) | orchestrator | event-driven | The recursive predict/cache + cold-start prior is novel; only the *manifest contract it consumes* has an analog. RESEARCH Code Examples; D-03/D-04. |
| `scripts/seg/backbones.py` (`Backbone` protocol, ViT reshape, timm DINOv2/Swin) | model | transform | No `Protocol`/declared-stride seam, no timm usage anywhere in repo (`timm` not even in `requirements.txt`). RESEARCH Pattern 5 + Code Examples; D-05. |
| Variant-B widened patch-embed (15-ch) for all 3 backbones | model | transform | Structural surgery with no precedent; construction-only (D-06a). CONTEXT D-03a Variant B. |

## Metadata

**Analog search scope:** `scripts/` (read `finetune_paligemma.py`, `tiling.py`;
grepped `build_dataset.py` for split schema), `tests/` (read `test_tiling.py`,
`conftest.py`, `test_split.py`; scanned `tests/integration/`), `requirements.txt`,
`pytest.ini`. Confirmed no `.claude/skills` or `.agents/skills`; no `CLAUDE.md`.
**Files scanned:** 9 read in full + directory listings of `tests/` and
`tests/integration/`.
**Pattern extraction date:** 2026-05-15
