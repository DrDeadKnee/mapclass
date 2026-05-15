"""
Orchestrate the synthetic dataset build from a directory of Azgaar GeoJSON exports.

Per resolved decision A4 (option (a)): each (Azgaar source × render style) is its
own canonical per-map directory ``<azgaar_id>__<style>/`` containing exactly one
``image.png`` (the S-style render) plus ``land_cover.png`` / ``topography.png``
(identical across styles for the same source — generated once and shared) plus
``sample_weights.json``.

v1 synthetic source target: **N=100 Azgaar source maps** (user-confirmed A7
2026-05-15; research recommended N=50, user locked N=100) — roughly ~15–16 maps
per template across ~12 continent templates, which makes a stratified 15%
hold-out statistically robust for EVAL-01. The split logic is N-agnostic: the
build does not hard-fail on fewer/more sources.

Sub-commands
------------
  build  — render + label every source into a seeded, stratified, frozen
           train/test split under ``<out-dir>/{train,test}/<id>__<style>/``.

Train/test split (D-15..D-18)
-----------------------------
  * Stratified by Azgaar continent *template* (D-16). Azgaar GeoJSON exports do
    NOT expose a heightmap-template field (confirmed against the Plan-01 conftest
    fixture; no raw exports on disk), so the template key falls back to a
    filename-derived prefix — see ``template_key``. **User awareness:** if real
    Azgaar exports later expose a template field, revisit this key BEFORE the
    first ``split.json`` is frozen.
  * Seeded with a fixed constant (``_SPLIT_SEED = 42``) for reproducibility (D-16).
  * ~15% of source maps held out, each template contributing proportionally (D-16).
  * Split is at the WHOLE Azgaar source-map level — ALL render styles of a
    held-out source go to ``test/`` (D-15). Zero cross-style leakage by
    construction.
  * FIRST build computes the split and WRITES ``<out-dir>/split.json`` listing
    the held-out test source-map IDs (D-18). EVERY subsequent build READS
    ``split.json`` and never recomputes/mutates it: IDs in the list → ``test/``,
    everything else (including newly generated sources) → ``train/`` (D-17, D-18).
    The Phase 4 test set never changes after the first recording.

Usage
-----
  python build_dataset.py build [--raw-dir RAW] [--out-dir OUT] [--styles ...]

Defaults:
  --raw-dir   data/synthetic/raw
  --out-dir   data/synthetic
  --styles    flat illustrated satellite
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import tiling
from label import make_label_arrays
from render import render_one
from synthetic_weights import write_sample_weights

_REQUIRED_MAP_FILES = (
    "image.png", "land_cover.png", "topography.png", "sample_weights.json",
)

_STYLE_SEP = "__"
_SPLIT_SEED = 42          # D-16 — fixed for reproducibility; never change.
_TEST_FRACTION = 0.15     # D-16 — ~15% stratified hold-out.
_SPLIT_FILENAME = "split.json"

# v1 synthetic source target — user-confirmed A7 (2026-05-15).
N_TARGET_V1 = 100         # ~15–16 maps per template across ~12 templates.


def _sanitize_stem(stem: str) -> str:
    """Sanitize an Azgaar source stem for safe filesystem path use (T-02-08)."""
    return re.sub(r"[^\w-]", "_", stem)


def template_key(src_id: str) -> str:
    """Derive the stratification (continent-template) key from a source ID.

    Azgaar GeoJSON exports do not carry a heightmap-template name, so the key
    is the source stem with its trailing numeric/index suffix stripped:
    ``europe_07`` / ``europe-7`` → ``europe``. A separator (``_``/``-``/
    space) is REQUIRED before the numeric suffix (WR-08) so a stem like
    ``m12``, ``world2`` or ``r12map`` is NOT split mid-token (the old
    optional-separator pattern grouped ``m12`` → ``m``).
    A source with no separator+trailing-digit suffix falls back to the
    whole sanitized stem (its own stratum), which keeps the splitter
    correct (never starves a singleton).
    """
    s = _sanitize_stem(src_id)
    m = re.match(r"^(.+?)[ _\-]+\d+$", s)
    key = m.group(1) if m and m.group(1) else s
    return key.strip("_-").lower() or s.lower()


def stratified_split(source_ids, seed: int = _SPLIT_SEED,
                      test_fraction: float = _TEST_FRACTION):
    """Return the sorted list of held-out (test) source IDs.

    Deterministic: stratifies by ``template_key`` and, within each template,
    seeds an independent RNG (``seed`` salted by the template name) so the
    hold-out is reproducible across re-invocations and stable as new sources
    are appended. Each template contributes ``round(n * test_fraction)`` (at
    least 1 when the template has >=1 source and the global fraction > 0).
    """
    by_template: dict[str, list[str]] = {}
    for sid in source_ids:
        by_template.setdefault(template_key(sid), []).append(sid)

    test_ids: list[str] = []
    for tmpl in sorted(by_template):
        members = sorted(by_template[tmpl])
        n = len(members)
        k = int(round(n * test_fraction))
        if k == 0 and n > 0 and test_fraction > 0:
            k = 1
        k = min(k, n)
        rng = random.Random(f"{seed}:{tmpl}")
        test_ids.extend(rng.sample(members, k))
    return sorted(test_ids)


def load_or_create_split(out_dir: Path, source_ids) -> set[str]:
    """Read a frozen ``split.json`` if present, else compute + write it once.

    D-18: the manifest is snapshotted at the FIRST build and is frozen by ID
    list thereafter — this function never recomputes or mutates an existing
    ``split.json``. Returns the set of test (held-out) source IDs.
    """
    split_path = Path(out_dir) / _SPLIT_FILENAME
    if split_path.exists():
        data = json.loads(split_path.read_text())
        return set(data["test"])

    # WR-08: the template_key heuristic is provisional. Print the derived
    # {template: [member_ids]} grouping BEFORE freezing split.json so a
    # human can sanity-check stratification (mis-grouping skews the
    # EVAL-01 hold-out proportions and is otherwise invisible).
    grouping: dict[str, list[str]] = {}
    for sid in source_ids:
        grouping.setdefault(template_key(sid), []).append(sid)
    print("  Derived stratification groups (review before split.json is "
          "frozen — WR-08):")
    for tmpl in sorted(grouping):
        print(f"    {tmpl}: {sorted(grouping[tmpl])}")

    test_ids = stratified_split(source_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(
        {
            "seed": _SPLIT_SEED,
            "test_fraction": _TEST_FRACTION,
            "test": test_ids,
        },
        indent=2,
    ))
    return set(test_ids)


def build_one_source(
    geojson_path: Path,
    split_root: Path,
    styles=("flat", "illustrated", "satellite"),
) -> list[Path]:
    """Build every per-(source×style) dir for a single Azgaar source.

    ``split_root`` is the train/ or test/ root the caller already routed this
    source into (D-17). ``land_cover.png`` / ``topography.png`` are rasterised
    once and saved byte-identically into each style dir (shared label, D-15 /
    A4 option (a)). Returns the list of created style directories.
    """
    geojson_path = Path(geojson_path)
    src_id = _sanitize_stem(geojson_path.stem)

    # Rasterise the shared labels exactly once for this source.
    lc_img, topo_img = make_label_arrays(geojson_path)

    created: list[Path] = []
    for style in styles:
        out = Path(split_root) / f"{src_id}{_STYLE_SEP}{style}"
        out.mkdir(parents=True, exist_ok=True)

        img = render_one(geojson_path, style)
        img.save(out / "image.png")
        lc_img.save(out / "land_cover.png")
        topo_img.save(out / "topography.png")
        write_sample_weights(out, geojson_path.stem)

        # Tile into nested pyramids. ``out`` already lives under the train/ or
        # test/ root the caller routed this whole source into, and the tiler
        # defaults to ``out/pyramids`` — so every pyramid of a held-out source
        # stays on the test/ side, never crossing the frozen split boundary
        # (EVAL-01 zero-leakage, T-02-15). Guard partial dirs (T-02-16).
        missing = [f for f in _REQUIRED_MAP_FILES if not (out / f).exists()]
        if missing:
            print(f"  SKIP tiling {out.parent.name}/{out.name}: missing "
                  f"{', '.join(missing)}")
        else:
            tiling.tile(out)

        created.append(out)
        print(f"  built {out.parent.name}/{out.name}  ({lc_img.width}x{lc_img.height}px)")
    return created


def build(raw_dir: str | Path, output_dir: str | Path,
          styles=("flat", "illustrated", "satellite")) -> None:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    geojsons = sorted(raw_dir.glob("*.geojson"))
    if not geojsons:
        print(f"No .geojson files found in {raw_dir}")
        sys.exit(1)

    n = len(geojsons)
    print(f"Found {n} source map(s) in {raw_dir} "
          f"(v1 target N={N_TARGET_V1}, ~15-16/template across ~12 templates)")
    if n < N_TARGET_V1:
        print(f"  note: {n} < N={N_TARGET_V1} v1 target — splitter is N-agnostic, "
              f"not hard-failing.")

    source_ids = [_sanitize_stem(g.stem) for g in geojsons]

    # CR-02: distinct raw stems can sanitize to the SAME src_id (e.g.
    # ``europe (1)`` and ``europe-1`` → ``europe_1``). That collapses both
    # sources into one ``<src_id>__<style>/`` dir (silent data loss) and,
    # worse, can route a colliding pair across the frozen train/test
    # boundary (EVAL-01 leakage). Fail loudly BEFORE any build/split.
    collisions: dict[str, list[str]] = {}
    for g, sid in zip(geojsons, source_ids):
        collisions.setdefault(sid, []).append(g.name)
    dupes = {sid: names for sid, names in collisions.items() if len(names) > 1}
    if dupes:
        print("FATAL: source stems collide after sanitization — rename the "
              "raw .geojson files; aborting to protect the frozen split "
              "(EVAL-01).")
        for sid in sorted(dupes):
            print(f"  {sid!r} <- {sorted(dupes[sid])}")
        sys.exit(1)

    test_ids = load_or_create_split(output_dir, source_ids)
    print(f"  split.json: {len(test_ids)} held-out test source(s) (frozen)")

    train_root = output_dir / "train"
    test_root = output_dir / "test"

    for geojson_path in geojsons:
        sid = _sanitize_stem(geojson_path.stem)
        dest = test_root if sid in test_ids else train_root
        print(f"\n--- {geojson_path.stem} -> {dest.name}/ ---")
        try:
            build_one_source(geojson_path, dest, styles=styles)
        except Exception as exc:  # per-source resilience (T-02-09)
            print(f"  FAILED {geojson_path.name}: {exc}")

    print(f"\nDone. Dataset written to {output_dir} (train/ + test/)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the synthetic (Azgaar + Pillow) training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--raw-dir", type=Path,
                       default=Path("data/synthetic/raw"),
                       help="Directory of Azgaar .geojson exports")
        p.add_argument("--out-dir", type=Path,
                       default=Path("data/synthetic"),
                       help="Dataset root (train/ + test/ + frozen split.json)")
        p.add_argument("--styles", nargs="+",
                       default=["flat", "illustrated", "satellite"],
                       help="Render styles; each becomes its own <id>__<style>/ dir")

    p_build = sub.add_parser(
        "build",
        help=(f"Render+label every source into a seeded stratified frozen "
              f"train/test split (v1 target N={N_TARGET_V1} Azgaar source "
              f"maps, ~15-16 per template across ~12 continent templates; "
              f"~15%% stratified hold-out, frozen split.json — D-15..D-18)"),
    )
    add_common(p_build)

    args = parser.parse_args()
    if args.command == "build":
        build(args.raw_dir, args.out_dir, tuple(args.styles))


if __name__ == "__main__":
    main()
