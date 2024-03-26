from pyawd import VectorAcousticWaveDataset2D, VectorAcousticWaveDataset3D
from tqdm.auto import tqdm
import numpy as np

def generate_interrogators_dataset(X):
    res = []
    for i in tqdm(range(X.size)):
        res.append(np.array([list(X[i][1].values())]))
    return np.array(res)

base_train = "./train2D"
N = 10000
nx = 32
interrogators = [(nx//4, 0), (-nx//4, 0)]
train = VectorAcousticWaveDataset2D(N, nx=nx, interrogators=interrogators, velocity_model="Marmousi", openmp=True)
train.save(base_train)

X = generate_interrogators_dataset(train)

np.save(base_train+"_interrogators_data.npy", X)
Y = np.array([train.get_epicenter(i)/(train.nx/2) for i in range(train.size)])
np.save(base_train+"_epicenters.npy", Y)

base_test = "./test2D"
N = 200
nx = 32
test = VectorAcousticWaveDataset2D(N, nx=nx, interrogators=interrogators, velocity_model="Marmousi", openmp=True)
test.save(base_test)

X_test = generate_interrogators_dataset(test)
np.save(base_test+"_interrogators_data.npy", X_test)

Y_test = np.array([test.get_epicenter(i)/(test.nx/2) for i in range(test.size)])
np.save(base_test+"_epicenters.npy", Y_test)
