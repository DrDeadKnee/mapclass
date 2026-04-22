"""
Fine-tune PaliGemma-3B on illustrated map hex tiles using LoRA.

Strategy to prevent forgetting:
  - Language model (Gemma) is completely frozen — base weights AND LoRA params.
  - LoRA adapters are applied ONLY to the SigLIP vision encoder attention layers.
  - The multi-modal projector is kept trainable (small; bridges adapted vision
    features into the frozen language model's token space).

This means the model's text understanding (including historical map text such as
"Alpes", "Sahara", "Mare Mediterraneum") is entirely preserved, while the vision
encoder learns to recognise illustrated hex-tile terrain aesthetics.

Requires:
  pip install torch transformers>=4.41 peft>=0.10 accelerate bitsandbytes pillow

Usage:
  python finetune_paligemma.py \\
      --toon-dir data/toons \\
      --output-dir checkpoints/paligemma-terrain \\
      [--model-id google/paligemma-3b-pt-224] \\
      [--epochs 2] \\
      [--batch-size 4] \\
      [--grad-accum 4] \\
      [--lr 1e-4] \\
      [--lora-rank 16] \\
      [--use-4bit]

The saved checkpoint is a PEFT adapter directory (model weights + adapter config)
that can be loaded alongside the base model at inference time.
"""

import argparse
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from augment import (
    BG_PARCHMENT,
    BG_WHITE,
    composite,
    make_grid,
    to_faded,
    to_grayscale,
    to_parchment,
)
from toon_mapping import (
    PROMPT_TEMPLATES,
    label_to_answer,
    load_labeled_tiles,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ToonDataset(Dataset):
    """
    Builds training samples from hex tile images by applying augmentations
    and pairing each image with a terrain-description prompt/answer.

    Each tile produces samples from four visual augmentations:
      original, grayscale, parchment, faded
    Plus one 2×2 composite per augmentation (same class, random variants).
    Each (image, aug) pair is combined with every prompt template.

    This yields roughly 119 tiles × 4 augs × 2 layouts × 5 prompts ≈ 4760 samples.
    Samples are shuffled at construction time.
    """

    AUGMENTATIONS = ["original", "grayscale", "parchment", "faded"]

    def __init__(
        self,
        toon_dir: str | Path,
        processor: PaliGemmaProcessor,
        use_grids: bool = True,
    ) -> None:
        self.processor = processor
        self.samples: list[tuple[Image.Image, str, str]] = []  # (image, prompt, answer)

        tiles = load_labeled_tiles(toon_dir)
        if not tiles:
            raise ValueError(f"No labeled tiles found in {toon_dir}")

        # Group tiles by (lc, topo) class for grid construction
        class_groups: dict[tuple, list[Path]] = {}
        for path, lc, topo in tiles:
            key = (lc, topo)
            class_groups.setdefault(key, []).append(path)

        for path, lc, topo in tiles:
            raw = Image.open(path)
            class_key = (lc, topo)
            class_peers = class_groups[class_key]

            for aug_name in self.AUGMENTATIONS:
                aug_img = self._apply_aug(raw, aug_name)
                layouts = [aug_img]

                if use_grids and len(class_peers) >= 2:
                    # 2×2 composite of same-class variants (random selection)
                    peer_imgs = [
                        self._apply_aug(Image.open(p), aug_name)
                        for p in random.sample(class_peers, min(4, len(class_peers)))
                    ]
                    layouts.append(make_grid(
                        peer_imgs, rows=2, cols=2,
                        bg=BG_PARCHMENT if aug_name == "parchment" else BG_WHITE,
                    ))

                for img in layouts:
                    for prompt_idx, prompt in enumerate(PROMPT_TEMPLATES):
                        answer = label_to_answer(lc, topo, description_idx=prompt_idx)
                        self.samples.append((img, prompt, answer))

        random.shuffle(self.samples)

    @staticmethod
    def _apply_aug(img: Image.Image, aug_name: str) -> Image.Image:
        if aug_name == "original":
            return composite(img)
        if aug_name == "grayscale":
            return to_grayscale(img)
        if aug_name == "parchment":
            return to_parchment(img)
        if aug_name == "faded":
            return to_faded(img)
        raise ValueError(f"Unknown augmentation: {aug_name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, prompt, answer = self.samples[idx]
        inputs = self.processor(
            images=image,
            text="<image>" + prompt,
            suffix=answer,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True,
        )
        # Remove batch dimension added by processor
        return {k: v.squeeze(0) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------

def apply_lora(model: PaliGemmaForConditionalGeneration, rank: int, alpha: int) -> None:
    """
    Apply LoRA adapters to the SigLIP vision encoder attention layers.
    The language model (Gemma) is completely frozen — both base weights and
    any LoRA parameters that peft might create there.

    SigLIP uses  out_proj  for the output projection.
    Gemma uses   o_proj    for the output projection.
    Targeting ["q_proj", "k_proj", "v_proj", "out_proj"] therefore hits SigLIP
    attention fully while leaving Gemma's o_proj untouched. q/k/v appear in
    both, but we freeze the Gemma ones explicitly after get_peft_model.
    """
    from peft import LoraConfig, get_peft_model

    # Step 1: freeze language model base weights
    model.model.language_model.requires_grad_(False)

    # Step 2: wrap model with LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    # Step 3: freeze any LoRA params that peft added inside the language model
    for name, param in peft_model.named_parameters():
        if "language_model" in name:
            param.requires_grad_(False)

    # Multi-modal projector stays trainable (bridges vision → language token space)
    for param in peft_model.base_model.model.model.multi_modal_projector.parameters():
        param.requires_grad_(True)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return peft_model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    # Load model and processor
    print(f"Loading {args.model_id} …")
    processor = PaliGemmaProcessor.from_pretrained(args.model_id)

    load_kwargs: dict = {"torch_dtype": dtype}
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id, **load_kwargs
    )

    model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_rank * 2)

    # Dataset and loader
    print("Building dataset …")
    dataset = ToonDataset(args.toon_dir, processor, use_grids=True)
    print(f"Dataset size: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # Optimizer and schedule
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    model.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type=device.type, dtype=dtype):
                outputs = model(**batch)

            step_loss = outputs.loss.item()
            loss = outputs.loss / args.grad_accum
            loss.backward()
            epoch_loss += step_loss

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                lr_now = scheduler.get_last_lr()[0]
                avg = epoch_loss / step
                writer.add_scalar("loss/step", step_loss, global_step)
                writer.add_scalar("loss/running_avg", avg, global_step)
                writer.add_scalar("lr", lr_now, global_step)
                writer.add_scalar("epoch", epoch, global_step)

                if global_step % 20 == 0:
                    print(f"  Epoch {epoch} step {global_step} | loss {avg:.4f} | lr {lr_now:.2e}")

        epoch_avg = epoch_loss / len(loader)
        writer.add_scalar("loss/epoch_avg", epoch_avg, epoch)
        print(f"Epoch {epoch} complete | avg loss {epoch_avg:.4f}")

        # Save checkpoint after each epoch
        ckpt = output_dir / f"epoch{epoch:02d}"
        model.save_pretrained(ckpt)
        processor.save_pretrained(ckpt)
        print(f"  Saved checkpoint → {ckpt}")

    writer.close()
    print(f"Training complete. Final checkpoint: {output_dir / f'epoch{args.epochs:02d}'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PaliGemma-3B on map hex tiles")
    parser.add_argument("--toon-dir",    default="data/toons",
                        help="Directory containing hex tile PNGs")
    parser.add_argument("--output-dir",  default="checkpoints/paligemma-terrain",
                        help="Where to save PEFT adapter checkpoints")
    parser.add_argument("--model-id",    default="google/paligemma-3b-pt-224",
                        help="HuggingFace model ID (must be the base pretrained variant)")
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--grad-accum",  type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = batch-size × grad-accum)")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--lora-rank",   type=int,   default=16,
                        help="LoRA rank r (alpha is set to 2r automatically)")
    parser.add_argument("--use-4bit",    action="store_true",
                        help="Load model in 4-bit (bitsandbytes) — reduces VRAM to ~4 GB")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
