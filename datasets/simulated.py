import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"
    __name__ = 'test'

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [
            (50, 20)
        ]
    }

    def __init__(self, n_samples=10, n_features=50, kernel_size=5,
                 signal_length=50, std_noise=0.1, sparsity=0.1,
                 random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.signal_length = signal_length
        self.noise = std_noise
        self.sparsity = sparsity
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        X = rng.normal(size=(self.n_features, self.kernel_size))

        theta = rng.random(size=(self.n_features,
                                 self.signal_length,
                                 self.n_samples))
        theta = theta > 1 - self.sparsity

        y = np.concatenate([np.array([np.convolve(theta_k, d_k, mode="same")
                                     for theta_k, d_k in zip(theta[:, :, i], X)]).sum(axis=0).reshape(-1, 1)
                            for i in range(theta.shape[2])], axis=1)
        y += rng.normal(scale=self.noise, size=y.shape)

        data = dict(X=X, y=y)

        return self.n_features, data
