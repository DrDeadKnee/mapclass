# Phase 1: End-to-End Skeleton (Synthetic + Mock Backbone) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in 01-CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-08
**Phase:** 01-end-to-end-skeleton
**Areas discussed:** Mock backbone fidelity, EVAL-03 protocol artifact format

---

## Mock Backbone Fidelity

### Q1.1 — What output should the Mock backbone produce?

| Option | Description | Selected |
|--------|-------------|----------|
| Random uniform per-pixel | Pure noise; eval NLL ~log(num_classes); zero diagnostic value | |
| Class-frequency prior | Outputs marginal class distribution; gives prior baseline NLL | |
| Tiny learnable conv stub | 1–2 conv layers RGB → features, ~10–100k params; full backprop path | ✓ |
| Hybrid: prior at init + learnable stub | Initialized to prior, with learnable head | |

**User's choice:** Tiny learnable conv stub
**Notes:** Mock should exercise the FULL training path (optimizer, scheduler, mixed precision, gradient flow), not just shape contracts. Phase 1 success includes "loss visibly decreases" — confirming end-to-end gradient flow before SmolVLM lands.

---

### Q1.2 — What feature-dict shape should Mock emit?

| Option | Description | Selected |
|--------|-------------|----------|
| Single stage, single tensor | `{"features": [B, C, H/p, W/p]}`; simplest | ✓ |
| Mirror SmolVLM stage names | `{"vision.layer_8": ..., "vision.layer_16": ..., ...}`; same head code works in Phase 2 | |
| 2–3 generic stages (low/mid/high) | Multi-stage but generic; adapter logic in Phase 2 | |

**User's choice:** Single stage, single tensor
**Notes:** Multi-stage skip-connection plumbing is deferred to Phase 2. Phase 1 stays MVP-simplest. This means Phase 2 plan must include a head-refactor scope when SmolVLM's multi-stage features land.

---

### Q1.3 — Should Mock's init be seeded for reproducibility?

| Option | Description | Selected |
|--------|-------------|----------|
| Seeded init from config | `training_seed` controls Mock's conv-stub init; deterministic | ✓ |
| Seeded init with override flag | Default seeded; `--no-seed` for stress testing | |
| Always seeded, never overrideable | Simplest deterministic-only path | |

**User's choice:** Seeded init from config
**Notes:** PITFALL 15 prevention from day one. No override flag — keeps the seeding utility uniformly applied across Phase 2+ backbones too.

---

### Q1.4 — How should a Mock-trained checkpoint identify itself?

| Option | Description | Selected |
|--------|-------------|----------|
| Metadata-only | `backbone: 'mock'` recorded; no enforcement | ✓ |
| Refuse at SHIP only | `is_ship_eligible: false`; Phase 6 packaging raises RuntimeError | |
| Refuse at SHIP + warn at infer load | Same as above + warning printed on every infer.load_model() | |

**User's choice:** Metadata-only
**Notes:** User asked to clarify what "hard refuse" meant. After clarification (refusal at packaging boundary, not at load), user chose the lightest-weight option. Discipline-only — single-author project, operator-friction cost of refusal logic judged not worth it.

---

## EVAL-03 Protocol Artifact Format

### Q2.1 — What format should the EVAL-03 protocol artifact take?

| Option | Description | Selected |
|--------|-------------|----------|
| Markdown only | Human-readable .md, discipline-enforced | ✓ |
| JSON + schema (programmatic) | Hash-comparison at compare time; harder to read | |
| Hybrid: protocol.md + protocol.lock.json | Most ceremony, most resilient | |

**User's choice:** Markdown only
**Notes:** `mapclass/configs/EVAL-03_protocol.md`, one page. The hash-assertion test in Phase 5 compares configs against each other (not against the protocol); the protocol is the audit trail, not the enforcement mechanism.

---

### Q2.2 — When should the EVAL-03 accept threshold be locked?

| Option | Description | Selected |
|--------|-------------|----------|
| Lock now (Phase 1) at 10% relative NLL | Pre-registered, no goalpost shifting | |
| Lock now (Phase 1) at 5% relative NLL | Stricter | |
| Lock now with explicit logic, threshold deferred | Structure committed; % is `DECIDE_AT_PHASE_5` placeholder | ✓ |
| Defer entirely to Phase 5 | Most flexible; PITFALL 2 prevention weakest | |

**User's choice:** Lock now with explicit logic, threshold deferred
**Notes:** Compromise picks structure now, number later. PITFALL 2 prevention is partial — discipline + Q2.4's script guard close the gap.

---

### Q2.3 — How strict is EVAL-03 fairness?

| Option | Description | Selected |
|--------|-------------|----------|
| Strict: only `backbone:` field differs | Pure controlled experiment | ✓ |
| Loose: data/loss/splits/aug locked, hyperparameters scale | Each backbone in its native operating envelope | |
| Documented-difference: every delta justified | Permitted Differences table | |

**User's choice:** Strict — only backbone field differs
**Notes:** ⚠ Known tension flagged for Phase 5 planner: STACK.md says PaliGemma needs NF4 + LoRA on a 24 GB GPU; SmolVLM's native input is 384 vs PaliGemma's 224. Strict fairness may make PaliGemma OOM or distort training. Phase 5 plan must include a one-batch dry-run gate before committing to strict fairness. If dry-run fails, this decision is re-opened in Phase 5.

---

### Q2.4 — How does the protocol enforce "threshold set before EVAL-03 runs"?

| Option | Description | Selected |
|--------|-------------|----------|
| Two-commit policy in protocol text | Discipline-only; git log audit | |
| Two-commit + script-level guard | `eval_03_compare.py` refuses if `DECIDE_AT_PHASE_5` literal is present | ✓ |
| Two-commit + script guard + git-history mtime check | Most paranoid; catches retroactive editing | |

**User's choice:** Two-commit + script-level guard
**Notes:** Mechanical block via literal sentinel string. Phase 5 plan must include "commit threshold value" as a discrete, named task that runs before the compare script.

---

## Claude's Discretion

User explicitly deferred two areas to Claude's defaults (multiSelect did not include them):

1. **Package migration strategy** — Default chosen: New `mapclass/` package alongside untouched `scripts/`. Synthetic and historical pipelines preserved as-is. Phase 1 planner can reopen if a clean reason emerges, but should not surface this to the user without cause.
2. **Skeleton-run dataset size + observability** — Default chosen: ~100 synthetic samples, `assert_sample_valid()` strict at startup once per run, custom `SampleContractError` exception loud-fail.

Additional defaults captured in CONTEXT.md `<decisions>` § "Claude's Discretion": eval report format, taxonomy hash algorithm, splits.json structure.

---

## Deferred Ideas

- **Multi-stage feature dict for Mock** — punted to Phase 2 (SmolVLM brings real multi-stage; head refactor in Phase 2 scope).
- **Calibration (reliability diagram + ECE) in eval report** — TS-6 is a Phase 2+ extension; Phase 1's eval JSON schema is forward-compatible.
- **Mock-checkpoint refusal at SHIP / infer load** — if mock-checkpoint accidents become a real problem during development, revisit in Phase 6 / milestone 2.
- **Permitted Differences table for EVAL-03** — explicitly rejected in Phase 1 (strict chosen). If Phase 5 dry-run reveals strict is infeasible, may negotiate at that time as a Phase-5 scope amendment.
- **Git-history mtime check for deferred-threshold pre-registration** — rejected; if retroactive editing becomes a problem, v1.x audit tooling.
