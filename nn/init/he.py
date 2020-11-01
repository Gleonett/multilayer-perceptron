import torch
import numpy as np


def he(n_in, n_out):
    dsp = np.sqrt(2 / n_in)
    W = torch.rand((n_out, n_in)).uniform_(0, dsp)
    b = torch.zeros(n_out)
    return W, b