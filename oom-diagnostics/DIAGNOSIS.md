# Phase 4 Wave 1 OOM — Diagnosis

**Date:** 2026-05-16
**VM:** 64295 MiB RAM (~62 GiB), 16 vCPU, **swap = 0 B (none configured)**
**Run monitored:** phase 4 wave 1 (plan 04-01) executor, worktree-isolated, under guardrails.

## Outcome

This monitored run **did NOT OOM** — it completed (3 commits, tests green) at a
peak of **17,495 MiB system used / ~15.6 GiB in a single `python` pytest
process**. It survived only because the executor was forced to (a) scope pytest
to the 3 plan test files and (b) skip `pip install -r requirements.txt`. Without
those guardrails this same workload OOM-kills.

## Root cause (two compounding drivers)

### 1. Code defect — full-pyramid autograd graph retention (primary)

`scripts/seg/train_utils.py::train_step_variant_a` / `train_step_variant_b`
accumulate `total_loss` across **all 21 pyramid tiles** and call a single
`total_loss.backward()` **after** the entire recursive c2f walk
(`train_utils.py:296,325,332`).

The prior cache *is* correctly detached (line 315–317, the documented T-04-03
fix), so there is no backprop-through-time. **But** every per-tile forward graph
(1×896², 4×448², 16×224² activations through the model) is summed into one
`total_loss` and kept alive simultaneously until the deferred backward. Peak
memory ≈ 21 tile graphs at once instead of 1.

Evidence: `mem-detail.log` shows the single pytest process for
`test_loss_decreases_on_overfit` sawtoothing 6 GB → 16 GB → 8 GB → 14 GB → 6 GB
(allocate full 21-tile graph → backward frees it → next of 5 overfit iterations
rebuilds it). That test alone took 114.8 s (`pytest-run.log`).

With a **stub backbone** this is ~15.6 GB. With a real backbone
(PaliGemma-3B / SigLIP / DINOv2 / Swin, per `requirements.txt`) the per-tile
activations are orders of magnitude larger → tens of GB → OOM even at 62 GiB.

**Fix:** backward per tile so only one tile's graph is live at a time:
```python
tile_loss = weighted_joint_loss(...)
tile_loss.backward()                 # grads accumulate in .grad
total_loss += float(tile_loss.detach())   # scalar for logging only
# ... after the walk:
optimizer.step()
```
This bounds peak to one tile's graph (~1/21 of current) with identical gradients
(sum-of-losses backward == sum of per-loss backwards into .grad).

### 2. Test-harness amplifier (why the *unguarded* run OOMs harder)

- `pytest.ini` has `testpaths = tests`. A bare `pytest` (what the executor runs
  by default) **also collects `test_seg_backbones.py` and
  `test_seg_recursive.py`**, which import `torch`/`transformers`/`timm` and
  instantiate multi-billion-parameter backbones on CPU — adds 12+ GB on top of
  driver #1.
- `pip install -r requirements.txt` (`torch`, `transformers>=5.8`,
  `bitsandbytes`, `accelerate`, `peft`, `timm`) is itself a multi-GB resident
  footprint during build/import.
- **Zero swap**: the kernel has no soft-landing. The instant RSS crosses the
  ceiling it OOM-kills the largest task — frequently sshd or the Claude process,
  so the failure looks like "the run died" rather than a clean MemoryError.

## Why the bigger VM "might still happen"

The resize raised the ceiling to ~62 GiB but changed nothing about driver #1 or
the zero-swap cliff. Scoped + no-install peaked at ~17.5 GiB and fit. An
unscoped full-suite run (real backbones × the 21-graph retention bug) can still
exceed 62 GiB and will OOM-kill instantly with no swap.

## Recommendations (in priority order)

1. **Fix `train_step_variant_a/b`** to backward per tile (above). Highest
   leverage; makes training viable on CPU and far cheaper on GPU.
2. **Add swap** (e.g. 16–32 GiB swapfile). Converts an instant OOM-kill into
   slowdown + a catchable `MemoryError`, and makes diagnosis possible.
3. **Scope CI/executor pytest** so unit waves don't collect the
   backbone-loading suites — e.g. pytest markers (`@pytest.mark.heavy`) and run
   `pytest -m "not heavy"` for non-GPU waves, or per-plan test path scoping in
   the plan's verification commands.
4. Keep the `ulimit -v` cap convention for any backbone/training command so a
   runaway dies as one process, not a machine-wide OOM.

## Artifacts (this directory)

| file | what |
|---|---|
| `mem-fast.log` | 1–2 s cadence: avail/free/used + biggest process, with run markers |
| `mem-detail.log` | 5 s cadence: top-15 RSS process table (shows the pytest cmdline) |
| `peak.log` | running high-water mark (6 GB → 17.5 GB climb visible) |
| `oom-events.log` | kernel OOM-kill extract + run markers (no kill this run) |
| `kernel-follow.log` | verbatim kernel ring buffer during the run |
| `pytest-run.log` | the executor's scoped pytest output + durations |
| `mem-sampler.sh` / `oom-watch.sh` | the (re-runnable) monitors |

To re-monitor a future run:
`setsid nohup bash oom-diagnostics/mem-sampler.sh >/dev/null 2>&1 & disown`
and likewise `oom-watch.sh`.
