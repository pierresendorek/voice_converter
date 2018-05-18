import numpy as np


class VectorSetNormalizationParams:

    def __init__(self, vector_dim):
        self.sum_vector = np.zeros([vector_dim])
        self.sum_sq_vector = np.zeros([vector_dim])
        self.n_samples_used = 0

    def use_also_this_vector_for_estimation(self, vector):
        self.sum_vector += vector
        self.sum_sq_vector += vector ** 2
        self.n_samples_used += 1


    def get_mu_sigma2(self):
        # Uses the equality
        # E((X - mu)^2) = E(X^2) - E(X)^2
        mu = self.sum_vector / self.n_samples_used
        sigma2 = self.sum_sq_vector / self.n_samples_used - mu ** 2
        return mu, np.abs(sigma2)



if __name__ == "__main__":

    vsn = VectorSetNormalizationParams(3)
    vsn.use_also_this_vector_for_estimation(np.ones([3]))
    vsn.use_also_this_vector_for_estimation(np.ones([3]) * 2)
    print(vsn.get_mu_sigma2())