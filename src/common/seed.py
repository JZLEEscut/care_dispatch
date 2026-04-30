from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch seeding is optional because some CPU-only / CI builds can hang on
    # torch.manual_seed. Enable it explicitly with CARE_DISPATCH_SEED_TORCH=1.
    if os.environ.get('CARE_DISPATCH_SEED_TORCH', '0') == '1':
        try:
            import torch
            torch.manual_seed(seed)
            if os.environ.get('CARE_DISPATCH_SEED_CUDA', '0') == '1' and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
