import random
import numpy as np
import torch
from .device import get_device


def set_seed(seed: int) -> None:
    device = get_device()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)
