"""
Orchestrate the synthetic dataset build from GCS-hosted Azgaar GeoJSON exports.

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

GCS-canonical persistence (D-06/D-17/D-18 REVERSED 2026-05-16, 02-02):
  * Raw GeoJSON + manifest.json are read from GCS (gs://.../data/synthetic/raw/).
  * split.json is stored at gs://.../data/synthetic/split.json (frozen on first
    build; subsequent builds read the frozen split and never recompute it).
  * Pyramid output streams directly to GCS (no canonical local copy; --local-ok
    required to override — Pitfall R-1 guard).
  * validate_manifest is the ONLY gate before split.json is frozen (RW-02).
  * --refreeze-split deletes the GCS split.json and forces recompute (Pitfall R-3).

Sub-commands
------------
  build  — render + label every source into a seeded, stratified, frozen
           train/test split under ``gs://.../data/synthetic/{train,test}/<id>__<style>/``.

Train/test split (D-15..D-18)
-----------------------------
  * Stratified by Azgaar continent *template* (D-16). Template is authoritative
    from manifest.json (id_to_template from validate_manifest); template_key()
    fallback is only reached when manifest lookup misses (unreachable in normal
    operation since validate_manifest hard-fails on any mismatch).
  * Seeded with a fixed constant (``_SPLIT_SEED = 42``) for reproducibility (D-16).
  * ~15% of source maps held out, each template contributing proportionally (D-16).
  * Split is at the WHOLE Azgaar source-map level — ALL render styles of a
    held-out source go to ``test/`` (D-15). Zero cross-style leakage by
    construction.
  * FIRST build computes the split and WRITES split.json to GCS (D-18). EVERY
    subsequent build READS split.json and never recomputes/mutates it (D-17, D-18).

Usage
-----
  python build_dataset.py build [--raw-dir RAW] [--out-dir OUT] [--styles ...]
                                [--local-ok] [--refreeze-split]

Defaults:
  --raw-dir   gs://mapclass-training-northeast1/data/synthetic/raw
  --out-dir   gs://mapclass-training-northeast1/data/synthetic
  --styles    flat illustrated satellite
"""

import argparse
import io
import json
import math
import random
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# GCS imports (lazy — same pattern as gcs_checkpoint.py lines 36-39)
# ---------------------------------------------------------------------------

try:
    import gcsfs  # type: ignore[import]
except ModuleNotFoundError:
    gcsfs = None  # type: ignore[assignment]

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import tiling
from gcs_io import _GCSWriter, validate_manifest, GCS_PROJECT, DATA_PREFIX, mark_build_complete, is_build_complete
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

# GCS canonical split.json location (D-18 REVERSED 2026-05-16).
_GCS_SPLIT_PATH = f"{DATA_PREFIX}/synthetic/split.json"

# GCS canonical raw prefix (bare, no gs://).
_GCS_RAW_PREFIX = f"{DATA_PREFIX}/synthetic/raw"


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


def stratified_split(
    source_ids,
    seed: int = _SPLIT_SEED,
    test_fraction: float = _TEST_FRACTION,
    id_to_template: "dict[str, str] | None" = None,
):
    """Return the sorted list of held-out (test) source IDs.

    Deterministic: stratifies by manifest template (id_to_template, from
    validate_manifest), falling back to ``template_key`` only when the manifest
    lookup misses. Within each template, seeds an independent RNG (``seed``
    salted by the template name) so the hold-out is reproducible across
    re-invocations and stable as new sources are appended. Each template
    contributes ``round(n * test_fraction)`` (at least 1 when the template has
    >=1 source and the global fraction > 0).

    Parameters
    ----------
    source_ids: iterable of sanitized source ID strings.
    seed: PRNG seed (D-16; default _SPLIT_SEED = 42).
    test_fraction: fraction held out (D-16; default _TEST_FRACTION = 0.15).
    id_to_template: manifest-authoritative {src_id: template} mapping from
        validate_manifest. If None, falls back to template_key() for all IDs.
    """
    if id_to_template is None:
        id_to_template = {}

    by_template: dict[str, list[str]] = {}
    for sid in source_ids:
        # Manifest is authoritative; template_key only an unreached fallback.
        tmpl = id_to_template.get(sid, template_key(sid))
        by_template.setdefault(tmpl, []).append(sid)

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


def load_or_create_split(
    fs,
    source_ids,
    id_to_template: "dict[str, str] | None" = None,
    refreeze: bool = False,
) -> set[str]:
    """Read a frozen ``split.json`` from GCS if present, else compute + write it once.

    D-18 REVERSED (2026-05-16): split.json lives at
    ``gs://mapclass-training-northeast1/data/synthetic/split.json`` and is frozen
    at the FIRST build. This function never recomputes or mutates an existing
    split.json unless ``refreeze=True`` (Pitfall R-3 escape hatch).

    Parameters
    ----------
    fs: gcsfs.GCSFileSystem instance (already instantiated by caller).
    source_ids: iterable of sanitized source ID strings.
    id_to_template: manifest-authoritative {src_id: template} mapping.
    refreeze: if True, delete the existing GCS split.json and recompute.

    Returns the set of test (held-out) source IDs.
    """
    if refreeze:
        if fs.exists(_GCS_SPLIT_PATH):
            fs.rm(_GCS_SPLIT_PATH)
            print("  --refreeze-split: deleted existing GCS split.json (Pitfall R-3)")

    if fs.exists(_GCS_SPLIT_PATH):
        data = json.loads(fs.cat(_GCS_SPLIT_PATH).decode())
        print(f"  split.json: reading frozen split from GCS (D-18)")
        return set(data["test"])

    # WR-08: the template grouping is from manifest (authoritative) or template_key
    # (fallback). Print the derived {template: [member_ids]} grouping BEFORE
    # freezing split.json so a human can sanity-check stratification.
    grouping: dict[str, list[str]] = {}
    eff_id_to_tmpl = id_to_template or {}
    for sid in source_ids:
        tmpl = eff_id_to_tmpl.get(sid, template_key(sid))
        grouping.setdefault(tmpl, []).append(sid)
    print("  Derived stratification groups (review before split.json is "
          "frozen — WR-08):")
    for tmpl in sorted(grouping):
        print(f"    {tmpl}: {sorted(grouping[tmpl])}")

    test_ids = stratified_split(
        source_ids, id_to_template=id_to_template
    )
    payload = json.dumps(
        {
            "seed": _SPLIT_SEED,
            "test_fraction": _TEST_FRACTION,
            "test": test_ids,
        },
        indent=2,
    ).encode()
    fs.pipe_file(_GCS_SPLIT_PATH, payload)
    print(f"  split.json: wrote {len(test_ids)} held-out test IDs to GCS (frozen)")
    return set(test_ids)


def build_one_source(
    fs,
    gcs_raw_prefix: str,
    geojson_fn: str,
    split_root: Path,
    split_name: str,
    src_id: str,
    styles=("flat", "illustrated", "satellite"),
    out_gcs_prefix: str = "",
) -> list[Path]:
    """Build every per-(source×style) dir for a single Azgaar source.

    ``split_root`` is the local train/ or test/ scratch dir the caller already
    routed this source into (D-17). Pyramid output streams to GCS via _GCSWriter.
    ``land_cover.png`` / ``topography.png`` are rasterised once and saved
    byte-identically into each style dir (shared label, D-15 / A4 option (a)).
    Returns the list of created local style directories.
    """
    # Download the GeoJSON from GCS to local scratch for rendering.
    raw_bytes = fs.cat(f"{gcs_raw_prefix}/{geojson_fn}")
    local_geojson = split_root / geojson_fn
    local_geojson.write_bytes(raw_bytes)
    geojson_path = local_geojson

    # Rasterise the shared labels exactly once for this source.
    lc_img, topo_img = make_label_arrays(geojson_path)

    created: list[Path] = []
    for style in styles:
        out = split_root / f"{src_id}{_STYLE_SEP}{style}"
        out.mkdir(parents=True, exist_ok=True)

        img = render_one(geojson_path, style)
        img.save(out / "image.png")
        lc_img.save(out / "land_cover.png")
        topo_img.save(out / "topography.png")
        write_sample_weights(out, geojson_path.stem)

        # Guard partial dirs (T-02-16).
        missing = [f for f in _REQUIRED_MAP_FILES if not (out / f).exists()]
        if missing:
            print(f"  SKIP tiling {split_name}/{out.name}: missing "
                  f"{', '.join(missing)}")
        else:
            # Pyramid output to GCS — stream via _GCSWriter (D-06 REVERSED).
            gcs_map_prefix = (
                f"{out_gcs_prefix}/{split_name}/{src_id}{_STYLE_SEP}{style}"
            )
            # OQ1: skip only if _BUILD_COMPLETE sentinel is present (Pitfall R-2).
            # A prefix whose objects exist but whose sentinel is absent is a
            # partial/aborted build — rebuild it from scratch.
            if is_build_complete(fs, gcs_map_prefix):
                print(f"  SKIP (complete) {split_name}/{out.name}")
                created.append(out)
                continue
            gcs_pyr_writer = _GCSWriter(fs, f"{gcs_map_prefix}/pyramids")
            tiling.tile(out, out_root=gcs_pyr_writer)
            # OQ1: write sentinel LAST after all pyramid objects are written (T-02-40).
            mark_build_complete(fs, gcs_map_prefix)

        created.append(out)
        print(f"  built {split_name}/{out.name}  ({lc_img.width}x{lc_img.height}px)")
    return created


def build(
    raw_dir: "str | Path",
    output_dir: "str | Path",
    styles=("flat", "illustrated", "satellite"),
    local_ok: bool = False,
    refreeze_split: bool = False,
) -> None:
    """Build the synthetic dataset from GCS raw inputs.

    Parameters
    ----------
    raw_dir: GCS URI or local path to the directory of Azgaar .geojson exports.
    output_dir: GCS URI or local path to the dataset root (train/ + test/).
    styles: render styles; each becomes its own <id>__<style>/ dir.
    local_ok: if True, allow non-gs:// output_dir (Pitfall R-1 guard override).
    refreeze_split: if True, delete GCS split.json and recompute (Pitfall R-3).
    """
    raw_dir_str = str(raw_dir)
    output_dir_str = str(output_dir)

    # Pitfall R-1 guard (T-02-12): require --local-ok for non-GCS output.
    if not output_dir_str.startswith("gs://") and not local_ok:
        print(
            "FATAL: --out-dir does not start with 'gs://' and --local-ok was "
            "not specified. To write locally (recreating the root-cause "
            "ephemeral-compute defect), pass --local-ok explicitly."
        )
        sys.exit(1)

    # Instantiate GCS filesystem (or fall back for local testing).
    if gcsfs is None:
        raise ImportError(
            "gcsfs is not installed. Install on the build VM with: "
            "pip install gcsfs"
        )
    fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)

    # Strip gs:// prefix for bare GCS paths (mirrors gcs_checkpoint.py idiom).
    raw_gcs = raw_dir_str.lstrip("gs://") if raw_dir_str.startswith("gs://") else ""
    out_gcs = output_dir_str.lstrip("gs://") if output_dir_str.startswith("gs://") else ""

    # --- List raw .geojson files from GCS ---
    if raw_dir_str.startswith("gs://"):
        all_objects = fs.ls(raw_gcs)
        # Extract basenames of .geojson files.
        gcs_filenames = [
            o.rsplit("/", 1)[-1]
            for o in all_objects
            if o.endswith(".geojson")
        ]
        if not gcs_filenames:
            print(f"No .geojson files found at {raw_dir_str}")
            sys.exit(1)

        # Read manifest.json from GCS.
        manifest_bytes = fs.cat(f"{raw_gcs}/manifest.json")
        manifest = json.loads(manifest_bytes.decode())
    else:
        # Local fallback (for testing with --local-ok + non-gs:// raw-dir).
        raw_path = Path(raw_dir_str)
        gcs_filenames = [p.name for p in sorted(raw_path.glob("*.geojson"))]
        if not gcs_filenames:
            print(f"No .geojson files found in {raw_dir_str}")
            sys.exit(1)
        manifest_path = raw_path / "manifest.json"
        if not manifest_path.exists():
            print(f"FATAL: manifest.json not found in {raw_dir_str}")
            sys.exit(1)
        manifest = json.loads(manifest_path.read_text())

    n = len(gcs_filenames)
    print(f"Found {n} source map(s) at {raw_dir_str} "
          f"(v1 target N={N_TARGET_V1}, ~15-16/template across ~12 templates)")
    if n < N_TARGET_V1:
        print(f"  note: {n} < N={N_TARGET_V1} v1 target — splitter is N-agnostic, "
              f"not hard-failing.")

    source_ids = [_sanitize_stem(fn[:-8]) for fn in gcs_filenames]

    # CR-02: distinct raw stems can sanitize to the SAME src_id. Fail BEFORE any
    # build/split to protect the frozen split and prevent EVAL-01 leakage.
    collisions: dict[str, list[str]] = {}
    for fn, sid in zip(gcs_filenames, source_ids):
        collisions.setdefault(sid, []).append(fn)
    dupes = {sid: names for sid, names in collisions.items() if len(names) > 1}
    if dupes:
        print("FATAL: source stems collide after sanitization — rename the "
              "raw .geojson files; aborting to protect the frozen split "
              "(EVAL-01).")
        for sid in sorted(dupes):
            print(f"  {sid!r} <- {sorted(dupes[sid])}")
        sys.exit(1)

    # RW-02: validate_manifest STRICTLY BEFORE load_or_create_split.
    # This is the ONLY gate before split.json is frozen (T-02-10).
    id_to_template = validate_manifest(gcs_filenames, manifest)

    # Load or create the frozen split (D-18 GCS-canonical).
    test_ids = load_or_create_split(
        fs,
        source_ids,
        id_to_template=id_to_template,
        refreeze=refreeze_split,
    )
    print(f"  split.json: {len(test_ids)} held-out test source(s) (frozen)")

    # Local scratch for reads (render/label consume local files; pyramids stream to GCS).
    import tempfile
    with tempfile.TemporaryDirectory(prefix="mapclass_build_") as tmp_scratch:
        scratch = Path(tmp_scratch)
        train_root = scratch / "train"
        test_root = scratch / "test"
        train_root.mkdir(parents=True, exist_ok=True)
        test_root.mkdir(parents=True, exist_ok=True)

        for fn, sid in zip(gcs_filenames, source_ids):
            is_test = sid in test_ids
            split_name = "test" if is_test else "train"
            local_split_root = test_root if is_test else train_root
            print(f"\n--- {fn[:-8]} -> {split_name}/ ---")
            try:
                build_one_source(
                    fs=fs,
                    gcs_raw_prefix=raw_gcs if raw_dir_str.startswith("gs://") else str(raw_dir_str),
                    geojson_fn=fn,
                    split_root=local_split_root,
                    split_name=split_name,
                    src_id=sid,
                    styles=styles,
                    out_gcs_prefix=out_gcs,
                )
            except Exception as exc:  # per-source resilience (T-02-09)
                print(f"  FAILED {fn}: {exc}")

    print(f"\nDone. Dataset written to {output_dir_str} (train/ + test/)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the synthetic (Azgaar + Pillow) training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument(
            "--raw-dir",
            type=str,
            default="gs://mapclass-training-northeast1/data/synthetic/raw",
            help="GCS URI (or local path with --local-ok) for Azgaar .geojson exports",
        )
        p.add_argument(
            "--out-dir",
            type=str,
            default="gs://mapclass-training-northeast1/data/synthetic",
            help="GCS URI (or local path with --local-ok) for dataset root",
        )
        p.add_argument(
            "--styles", nargs="+",
            default=["flat", "illustrated", "satellite"],
            help="Render styles; each becomes its own <id>__<style>/ dir",
        )
        p.add_argument(
            "--local-ok",
            action="store_true",
            default=False,
            help=(
                "Allow non-gs:// --out-dir (overrides Pitfall R-1 guard). "
                "Use only for local testing — production builds must use GCS."
            ),
        )
        p.add_argument(
            "--refreeze-split",
            action="store_true",
            default=False,
            help=(
                "Delete the GCS split.json and recompute from current sources. "
                "WARNING: this changes the EVAL-01 test set — use only when "
                "the full Azgaar source set has been regenerated (Pitfall R-3)."
            ),
        )

    p_build = sub.add_parser(
        "build",
        help=(f"Render+label every source into a seeded stratified frozen "
              f"train/test split (v1 target N={N_TARGET_V1} Azgaar source "
              f"maps, ~15-16 per template across ~12 continent templates; "
              f"~15%% stratified hold-out, frozen GCS split.json — D-15..D-18)"),
    )
    add_common(p_build)

    args = parser.parse_args()
    if args.command == "build":
        build(
            args.raw_dir,
            args.out_dir,
            tuple(args.styles),
            local_ok=args.local_ok,
            refreeze_split=args.refreeze_split,
        )


if __name__ == "__main__":
    main()
