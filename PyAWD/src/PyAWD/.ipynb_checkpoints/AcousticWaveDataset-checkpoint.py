# PyAWD - AcousticWaveDataset
# Tribel Pascal - pascal.tribel@ulb.be

import numpy as np
import devito as dvt
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm

from PyAWD.GenerateVideo import generate_video
from PyAWD.utils import *
from PyAWD.Marmousi import *

dvt.logger.set_log_level('WARNING')

def solve_pde(grid, nx, ndt, ddt, epicenter, velocity_model):
    """
    Solves the Acoustic Wave Equation for the input parameters
    Arguments:
        - grid: a Devito Grid Object
        - nx: the discretization size of the array
        - ndt: the number of iteration for which the result is stored
        - ddt: the time step used for the Operator solving iterations
        - epicenter: the epicenter of the Ricker Wavelet at the beginning of the simulation
        - velocity_model: the velocity field across which the wave propagates
    Returns:
        - u: a Devito TimeFunction containing the solutions for the `ndt` steps
    """
    u = dvt.TimeFunction(name='u', grid=grid, space_order=2, save=ndt, time_order=2)
    u.data[:] = get_ricker_wavelet(nx, x0=epicenter[0], y0=epicenter[1])
    eq = dvt.Eq(u.dt2, (velocity_model**2)*(u.dx2+u.dy2))
    stencil = dvt.solve(eq, u.forward)
    op = dvt.Operator(dvt.Eq(u.forward, stencil), opt='noop')
    op.apply(dt=ddt)
    return u

class AcousticWaveDataset(torch.utils.data.Dataset):
    """
    A Pytorch dataset containing acoustic waves propagating in the Marmousi velocity field.
    Arguments:
        - size: the number of samples to generate in the dataset
        - nx: the discretization size of the array (maximum size is currently 955)
        - ddt: the time step used for the Operator solving iterations
        - dt: the time step used for storing the wave propagation step (this should be higher than ddt)
        - t: the simulations duration
    """
    def __init__(self, size, nx=128, ddt=0.01, dt=2, t=10):
        try:
            if dt < ddt:
                raise ValueError('dt should be >= ddt')
            self.size = size
            self.nx = min(nx, 955)
            self.ddt = ddt
            self.dt = dt
            self.nt = int(t/self.dt)
            self.ndt = int(self.nt*(self.dt/self.ddt))
            
            self.grid = dvt.Grid(shape=(self.nx, self.nx), extent=(1., 1.))
            self.velocity_model = dvt.Function(name='c', grid=self.grid)
            self.velocity_model.data[:] = Marmousi(self.nx).get_data()
    
            self.epicenters = torch.randint(-self.nx//2, self.nx//2, size=(self.size, self.size)).reshape((self.size, self.size))
    
            self.cmap = get_black_cmap()
            
            self.generate_data()
        except ValueError as err:
            print(err)

    def generate_data(self):
        """
        Generates the dataset content by solving the Acoustic Wave PDE for each of the `epicenters`
        """
        self.data = []
        for i in tqdm(range(self.size)):
            self.data.append(solve_pde(self.grid, self.nx, self.ndt, self.ddt, self.epicenters[i], self.velocity_model))

    def plot_item(self, idx):
        """
        Plots the simulation of the idx^th sample
        Arguments:
            - idx: the number of the sample to plot
        """
        epicenter, item = self[idx]
        fig, ax = plt.subplots(1, self.nt, figsize=(self.nt*3, 3))
        for i in range(self.nt):
            ax[i].imshow(self.velocity_model.data, vmin=np.min(self.velocity_model.data), vmax=np.max(self.velocity_model.data), cmap="gray")
            x = ax[i].imshow(item[i*(item.shape[0]//self.nt)], 
                             vmin=-np.max(np.abs(item[i*(item.shape[0]//self.nt):])), 
                             vmax=np.max(np.abs(item[i*(item.shape[0]//self.nt):])), 
                             cmap=self.cmap)
            ax[i].set_title("t = " + str(i*(item.shape[0]//self.nt)*self.dt) + "s")
            ax[i].axis("off")
            fig.colorbar(x)
        plt.tight_layout()
        plt.show()

    def generate_video(self, idx, filename, nb_images):
        """
        Generates a video representing the simulation of the idx^th sample propagation
        Arguments:
            - idx: the number of the sample to simulate in the video
            - filename: the name of the video output file (without extension)
                        The video will be stored in a file called `filename`.mp4
            - nb_images: the number of frames used to generate the video
        """
        u = solve_pde(self.grid, self.nx, self.ndt, self.ddt, self.epicenters[idx], self.velocity_model)
        generate_video(u.data[::self.ndt//(nb_images)], filename, dt=self.ndt*self.ddt/(nb_images), c=self.velocity_model, verbose=True)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.epicenters, self.data[idx].data[::int(self.ndt/self.nt)]