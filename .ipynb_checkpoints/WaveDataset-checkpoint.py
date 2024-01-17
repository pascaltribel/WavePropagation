from pde import CartesianGrid, WavePDE, ScalarField, MemoryStorage, movie_scalar, FieldCollection, DataTracker
import matplotlib.pyplot as plt
import numpy as np
import torch

class WaveDataset(torch.utils.data.Dataset):
    """
    Offers tools to use a dataset representing 2d-wave propagation from various epicenters in a given area.
    This area is a squared plane of side length 1 where each point has a given speed of propagation. It is stored in the `constraint` attribute.
    The speed of propagation is computed by the Finite Difference Method, implemented in the PyPDE package.
    """
    def __init__(self, num_samples, nx, speed, dt, t, interval, sampling_rate=1, verbose=False):
        """
        Attributes:
        - num_samples: the number of samples containted in the dataset
        - nx: the number of points determining the plane meshgrid
        - the wave propagation speed when no obstacle is met
        - dt: the step used between each FDM step
        - t: the duration of each propagation simulation
        - interval: the number of equally-spaced samples to take in each propagation experiment 
        - sampling_rate (1): the resolution proportion of the items obtained (max: 1) to allow subsampling
        - verbose (False): if True, displays a progress bar when generating the samples
        """
        self.num_samples = num_samples
        self.nx = nx
        self.speed = speed
        self.dt = dt
        self.t = t
        self.interval = interval
        self.sampling_rate = sampling_rate
        self.verbose = verbose
        self.flattened = False
        self.generate_constraint()
        self.generate_epicenters()
        self.generate_final_states()

    def generate_constraint(self, constraint=None):
        """
        If constraint is None, then a nx*nx matrix is constructed.
        It is made of a centered square of size nx//2 with a hole of size 1px on the middle of the right side, where the speed propagation is 0.
        The rest of the area has a wave propagation speed determined by the `speed` attribute.

        If constraint is not None, then it should be a custom constraint matrix of size nx*nx. 
        """
        if not constraint:
            self.constraint = np.full((self.nx, self.nx), self.speed)
            
            center = self.nx // 2
            side_length = self.nx // 2
        
            start_row, end_row = center - side_length // 2, center + side_length // 2 + 1
            start_col, end_col = center - side_length // 2, center + side_length // 2 + 1
        
            self.constraint[center - side_length // 2:1+center - side_length // 2, start_col:end_col] = 0
            self.constraint[center + side_length // 2:1+center + side_length // 2, start_col:end_col] = 0
            self.constraint[start_row:end_row, center - side_length // 2:1+center - side_length // 2] = 0
            self.constraint[start_row:end_row, center + side_length // 2:1+center + side_length // 2] = 0
            
            self.constraint[center:, center] = self.speed
        else:
            self.constraint = constraint
        
    def generate_epicenters(self):
        """
        Generates randomly `num_samples` epicenters. Each coordinate is between 0 and 1.
        """
        self.epicenters = torch.rand((self.num_samples, 2))

    def generate_final_states(self):
        """
        Generates the simulations for each of the `num_samples` epicenters.
        The `initial_densities` contains `num_samples` matrices (shape: `nx`, `nx`) containing the initial density of each simulation.
        The `final_states` attribute contains `num_samples` matrices (shape: `interval`, `nx`, `nx`) where the ith one is the ith state in the propagation simulation.
        """
        self.initial_densities = []
        self.final_states = []
        for idx in range(self.num_samples):
            print("Generating sample", idx, "/", self.num_samples)
            grid = CartesianGrid([[0, 1], [0, 1]], [self.nx, self.nx], periodic=[True, True])
            self.initial_densities.append(ScalarField(grid))
            self.initial_densities[-1].insert([self.epicenters[idx][0], self.epicenters[idx][1]], 1)
            data_tracker = DataTracker(lambda x: x.data[0].copy(), interval=self.t/(self.interval*self.dt))
            trackers = [data_tracker]
            if self.verbose:
                trackers.append("progress")
            eq = WavePDE(self.constraint.T)
            initial_condition = eq.get_initial_condition(self.initial_densities[idx])
            eq.solve(initial_condition, t_range=self.t/self.dt, dt=self.dt, tracker=trackers, adaptive=True)
            self.final_states.append(data_tracker.data[1:])
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a tuple:
        - The epicenter of the wave
        - The initial density        
        - The simulation
        """
        if not self.flattened:
            return self.epicenters[idx], self.initial_densities[idx].data[::int(1/self.sampling_rate), ::int(1/self.sampling_rate)], [i[::int(1/self.sampling_rate), ::int(1/self.sampling_rate)] for i in self.final_states[idx]]
        else:
            return self.epicenters[idx], self.initial_densities[idx].data[::int(1/self.sampling_rate), ::int(1/self.sampling_rate)].flatten(), np.array([i[::int(1/self.sampling_rate), ::int(1/self.sampling_rate)].flatten() for i in self.final_states[idx]])

    def generate_animation(self, idx, filename):
      """
      Generates an animation of the simulation for the idx^th sample
      """
      storage = MemoryStorage()
      eq = WavePDE(self.constraint.T)
      initial_condition = eq.get_initial_condition(self.initial_densities[idx])
      result = eq.solve(initial_condition, t_range=self.t/self.dt, dt=self.dt, tracker=["progress", storage.tracker(1, transformation=lambda x: x["u"])])
      movie_scalar(storage, filename=filename, progress=self.verbose)