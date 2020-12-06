import torch
import numpy as np


def xavier(n_in, n_out):
    dsp = np.sqrt(6 / (n_in + n_out))
    W = torch.rand((n_out, n_in)).uniform_(-dsp, dsp)
    b = torch.rand(n_out).uniform_(-dsp, dsp)
    return W, b
