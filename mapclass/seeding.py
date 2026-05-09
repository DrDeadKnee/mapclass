"""
Project-wide seeding utility (PITFALL 15 prevention; CONCERNS.md 8a mitigation).

Single seed value, no --no-seed override flag (CONTEXT.md D-03). Called at every
entry point: mapclass.train, mapclass.eval, mapclass.infer (test mode).

Pins random, numpy, torch CPU+CUDA, cudnn deterministic + benchmark off, and
PYTHONHASHSEED. Documented cost: ~10% throughput hit from cudnn.deterministic.
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed every RNG that affects training/inference reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
