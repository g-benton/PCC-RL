import math
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel

def expected_improvement(pred_dist, current_max, explore=0.):
    """
    Standard expected improvement
    take in predictive distribution and current max (optional exploration parameter)
    returns the expected improvment at all points
    """

    means = pred_dist.mean
    vars = pred_dist.variance

    std_vals = (means - current_max - explore)
    std_normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    first_term = std_vals.mul(std_normal.cdf(std_vals))
    second_term = vars.mul(std_normal.log_prob(std_vals).exp())

    return first_term + second_term

def max_mean(pred_dist, current_max):
    return pred_dist.mean


class Surrogate(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(Surrogate, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BayesOpt(object):
    """
    This is the class wrapper for Bayesian Optimization containing:
        the surrogate reward model (Gpytorch GP)
        the trainer for the surrogate
        the acquisition function
        the observed data


    Note that one issue in this work is that we have integer-valued input data
    Current implementation uses a naive approach that just rounds the acquired
    value to the nearest integer
    """
    def __init__(self, train_x=None, train_y=None, kernel=RBFKernel,
                 acquistion=expected_improvement, normalize=True, max_x=1000):

        self.acquisition = acquistion
        self.train_x = train_x.float().clone()
        self.max_x = max_x
        self.train_y = train_y.float().clone()

        self.normalize = normalize
        if self.normalize:
            self.train_x = self.train_x.div(self.max_x)

            # self.y_mean = self.train_y.mean()
            # self.y_std = self.train_y.std()
            # self.train_y = (self.train_y - self.y_mean).div(self.y_std)

        self.surrogate_lh = gpytorch.likelihoods.GaussianLikelihood()
        self.surrogate_lh.noise.data[0] = -1.
        self.surrogate = Surrogate(self.train_x, self.train_y, self.surrogate_lh,
                                   kernel=kernel)

    def train_surrogate(self, lr=0.01, iters=50):
        self.surrogate.train()
        self.surrogate_lh.train()

        optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.surrogate_lh,
                                                       self.surrogate)

        for i in range(iters):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.surrogate(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

    def acquire(self, **kwargs):
        """
        This will run the acquisition for the bayesian optimization method

        Right now it's fixed to just pick the best integer in [0, 1000],
        but further down the line we should figure out a way to get a viable
        range for CWND sizes.
        """
        test_points = torch.linspace(0, self.max_x, 1000).float()
        if self.normalize:
            int_test_points = test_points.clone()
            test_points = test_points.div(self.max_x)

        self.surrogate.eval()
        self.surrogate_lh.eval()

        test_dist = self.surrogate_lh(self.surrogate(test_points))
        best_y, ind = self.train_y.max(0)

        acquisition = self.acquisition(test_dist, best_y, **kwargs)
        best_ac, ind = acquisition.max(0)
        # print(test_points)
        print(test_points[ind])
        if self.normalize:
            return int_test_points[ind]
        else:
            return test_points[ind]

    def update_obs(self, x, y):
        self.train_y = torch.cat((self.train_y, y))
        if self.normalize:
            self.train_x = torch.cat((self.train_x, x.div(self.max_x)))
        else:
            self.train_x = torch.cat((self.train_x, x))


        self.surrogate.set_train_data(self.train_x, self.train_y, strict=False)
