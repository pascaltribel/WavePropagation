from PyAWD.AcousticWaveDataset import *
import torch

torch.set_default_device("cpu")

N = 250
nx = 256
train_small = AcousticWaveDataset(N, nx=nx)
torch.save(train_small, "../local/train_small.pt")

test_small = AcousticWaveDataset(N//5, nx=nx)
torch.save(test_small, "../local/test_small.pt")