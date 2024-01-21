#!/usr/bin/env python
# coding: utf-8

from WaveDataset import *
from pde import CartesianGrid, WavePDE, ScalarField, MemoryStorage, movie, FieldCollection, DataTracker
import matplotlib.pyplot as plt
import numpy as np
import torch


GENERATING = True
VERBOSE = True


NX = 2**6
SPEED = 0.002
DT = 0.01
T = 2**3
interval = 2**5
num_samples = 2**10
batch_size = 2**5
sampling_rate = 1

if GENERATING:
    dataset = WaveDataset(num_samples, NX, SPEED, DT, T, interval, sampling_rate=sampling_rate, verbose=VERBOSE)
    torch.save(dataset, './train.pt')
else:
    dataset = torch.load('./train.pt')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#center, inputs, outputs = dataset[0]
#plt.imshow(inputs, vmin=np.min(outputs), vmax=-np.min(outputs))
#plt.colorbar()
#plt.show()
#for i in range(interval):
#    plt.imshow(outputs[i], vmin=np.min(outputs), vmax=-np.min(outputs))
#    plt.colorbar()
#    plt.show()