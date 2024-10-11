import numpy as np
import torch
import random


def set_seed(seed: int = 137):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
