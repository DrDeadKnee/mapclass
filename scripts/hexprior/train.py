"""
Train the masked-hex terrain prior (Phase B).

Regions (npz files from build_hex_dataset.py) are split into train/val by
name; validation regions are fully held out so the reported KL measures
generalisation to unseen geography, not unseen masks.

Usage:
  python scripts/hexprior/train.py \
      --data gs://mapclass-training-northeast1/data/hexprior \
      --val-regions iberia \
      --steps 20000 --out runs/hexprior

--data and --out accept local paths or gs:// prefixes.
"""

import argparse
import glob
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexprior.dataset import (
    DEFAULT_MASK_RANGE,
    DEFAULT_WINDOW_SIZE,
    HexWindowDataset,
)
from hexprior.model import HexPriorModel, masked_kl_loss, masked_top1_accuracy


def list_region_paths(data: str) -> list[str]:
    if data.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        return sorted(f"gs://{p}" for p in fs.glob(data.rstrip("/") + "/*.npz"))
    return sorted(glob.glob(str(Path(data) / "*.npz")))


def _write_bytes(dest: str, payload: bytes) -> None:
    if dest.startswith("gs://"):
        import gcsfs
        gcsfs.GCSFileSystem().pipe_file(dest[len("gs://"):], payload)
    else:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(payload)


def save_checkpoint(dest: str, model: torch.nn.Module, step: int, config: dict) -> None:
    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "step": step, "config": config}, buf)
    _write_bytes(dest, buf.getvalue())


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 50) -> tuple[float, float]:
    model.eval()
    kls, accs = [], []
    for b, batch in enumerate(loader):
        if b >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["soft"], batch["xy"], batch["mask"], batch["pad"])
        kls.append(masked_kl_loss(logits, batch["soft"], batch["mask"]).item())
        accs.append(masked_top1_accuracy(logits, batch["soft"], batch["mask"]).item())
    model.train()
    return float(np.mean(kls)), float(np.mean(accs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="gs://mapclass-training-northeast1/data/hexprior")
    p.add_argument("--val-regions", nargs="*", default=["iberia"],
                   help="region name prefixes held out for validation")
    p.add_argument("--out", default="runs/hexprior")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW_SIZE)
    p.add_argument("--mask-lo", type=float, default=DEFAULT_MASK_RANGE[0])
    p.add_argument("--mask-hi", type=float, default=DEFAULT_MASK_RANGE[1])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1_000)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=1_000)
    args = p.parse_args()

    paths = list_region_paths(args.data)
    if not paths:
        raise SystemExit(f"no region npz files under {args.data}")
    stem = lambda path: Path(path).name  # noqa: E731
    val_paths = [q for q in paths
                 if any(stem(q).startswith(v) for v in args.val_regions)]
    train_paths = [q for q in paths if q not in val_paths]
    print(f"{len(train_paths)} train regions, {len(val_paths)} val regions")
    if not train_paths or not val_paths:
        raise SystemExit("need at least one train and one val region")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    mask_range = (args.mask_lo, args.mask_hi)
    train_ds = HexWindowDataset(train_paths, args.window, mask_range,
                                epoch_len=args.steps * args.batch, seed=args.seed)
    val_ds = HexWindowDataset(val_paths, args.window, mask_range,
                              epoch_len=50 * args.batch, seed=10_000_019)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              num_workers=args.workers, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers)

    model = HexPriorModel(d_model=args.d_model, n_heads=args.heads,
                          n_layers=args.layers).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / args.warmup)
        * 0.5 * (1.0 + np.cos(np.pi * min(1.0, s / args.steps))),
    )

    config = vars(args) | {"n_params": n_params}
    _write_bytes(f"{args.out.rstrip('/')}/config.json",
                 json.dumps(config, indent=2, default=str).encode())

    best_val = float("inf")
    t0 = time.time()
    model.train()
    for step, batch in enumerate(train_loader):
        if step >= args.steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["soft"], batch["xy"], batch["mask"], batch["pad"])
        loss = masked_kl_loss(logits, batch["soft"], batch["mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            rate = (step + 1) * args.batch / (time.time() - t0)
            print(f"step {step:6d}  kl {loss.item():.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {rate:.1f} windows/s",
                  flush=True)
        if step > 0 and step % args.val_every == 0:
            val_kl, val_acc = evaluate(model, val_loader, device)
            print(f"step {step:6d}  VAL kl {val_kl:.4f}  top1 {val_acc:.3f}",
                  flush=True)
            save_checkpoint(f"{args.out.rstrip('/')}/last.pt", model, step, config)
            if val_kl < best_val:
                best_val = val_kl
                save_checkpoint(f"{args.out.rstrip('/')}/best.pt", model, step, config)

    val_kl, val_acc = evaluate(model, val_loader, device)
    print(f"final  VAL kl {val_kl:.4f}  top1 {val_acc:.3f}")
    save_checkpoint(f"{args.out.rstrip('/')}/last.pt", model, args.steps, config)
    if val_kl < best_val:
        save_checkpoint(f"{args.out.rstrip('/')}/best.pt", model, args.steps, config)


if __name__ == "__main__":
    main()
