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
        ],
        "sparsity": [0.01, 0.005]
    }

    def __init__(self, n_samples=3, n_atoms=5, kernel_size=5,
                 signal_length=1024, std_noise=0.1, sparsity=0.005,
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
        """Return the data to run CSC on.

        Returns
        -------
        D : array, shape (n_atoms, kernel_size)
            Dictionary for the CSC problem.
        y : array, shape (n_times, n_samples)
            Signals to run the CSC on.
        """
        rng = np.random.RandomState(self.random_state)
        D = rng.normal(size=(self.n_atoms, self.kernel_size))

        theta = rng.random(size=(self.n_atoms,
                                 self.signal_length - self.kernel_size + 1,
                                 self.n_samples))
        theta *= theta < self.sparsity

        y = np.concatenate([
            np.sum([
                np.convolve(theta_k, d_k, mode="full")
                for theta_k, d_k in zip(theta[:, :, i], D)
            ], axis=0)[:, None]
            for i in range(theta.shape[2])
        ], axis=1)
        y += rng.normal(scale=self.noise, size=y.shape)

        return dict(D=D, y=y)
