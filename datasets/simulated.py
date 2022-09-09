import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"
    __name__ = 'test'

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_atoms': [
            (1, 20)
        ]
    }

    def __init__(self, n_samples=3, n_atoms=5, kernel_size=5,
                 signal_length=1024, std_noise=0.1, sparsity=0.01,
                 random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_atoms = n_atoms
        self.kernel_size = kernel_size
        self.signal_length = signal_length
        self.noise = std_noise
        self.sparsity = sparsity
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        D = rng.normal(size=(self.n_atoms, self.kernel_size))

        theta = rng.random(size=(self.n_atoms,
                                 self.signal_length - self.kernel_size + 1,
                                 self.n_samples))
        theta = theta > 1 - self.sparsity

        y = np.concatenate([
            np.sum([
                np.convolve(theta_k, d_k, mode="full")
                for theta_k, d_k in zip(theta[:, :, i], D)
            ], axis=0).reshape(-1, 1)
            for i in range(theta.shape[2])
        ], axis=1)
        y += rng.normal(scale=self.noise, size=y.shape)

        data = dict(D=D, y=y)

        return theta.shape, data
