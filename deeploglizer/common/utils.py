import torch
import random
import os
import numpy as np


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_device(device=-1):
    if device != -1 and torch.cuda.is_available():
        device = torch.device("cuda: " + str(device))
    else:
        device = torch.device("cpu")
    return device