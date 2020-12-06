import torch
import numpy as np


def standard(n_in, n_out):
    dsp = 1 / np.sqrt(n_out)
    W = torch.rand((n_out, n_in)).uniform_(-dsp, dsp)
    b = torch.rand(n_out).uniform_(-dsp, dsp)
    return W, b
