import math
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel
import numpy as np
import matplotlib.pyplot as plt

def expected_improvement(bayesopt, test_points, explore=0.01):
    """
    Standard expected improvement
    take in predictive distribution and current max (optional exploration parameter)
    returns the expected improvment at all points
    """
    bayesopt.surrogate.train()
    best = bayesopt.train_y.max()
    mu_sample = bayesopt.surrogate(bayesopt.train_x).mean
    current_max, max_ind = mu_sample.max(0)

    bayesopt.surrogate.eval()
    bayesopt.surrogate_lh.eval()
    pred_dist = bayesopt.surrogate_lh(bayesopt.surrogate(test_points))
    mu_test = pred_dist.mean
    var_test = pred_dist.variance.pow(0.5)


    imp = mu_test - current_max - explore
    z = imp.div(var_test)
    std_normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    ei = imp * std_normal.cdf(z) + var_test * std_normal.log_prob(z).exp()
    ei[var_test == 0.] = 0.

    # fig, ax1 = plt.subplots()
    # ax1.plot(bayesopt.train_x, bayesopt.train_y.detach(),
    #          marker='.', linestyle="None", label="observed")
    # ax1.plot(bayesopt.train_x, mu_sample.detach(), label="Train mean")
    # ax1.plot(test_points, mu_test.detach(), label="Pred Mean")
    # ax2 = ax1.twinx()
    # ax2.plot(test_points, ei.detach(),
    #          label="EI")
    # # ax1.set_xlim(0, 1.)
    # fig.legend()
    # plt.show()


    return ei

def MPI(bayesopt, test_points, explore=0.):
    bayesopt.surrogate.train()

    mu_sample = bayesopt.surrogate(bayesopt.train_x).mean
    current_max, max_ind = mu_sample.max(0)

    bayesopt.surrogate.eval()
    bayesopt.surrogate_lh.eval()


    pred_dist = bayesopt.surrogate_lh(bayesopt.surrogate(test_points))
    mu_test = pred_dist.mean
    std_test = pred_dist.variance.pow(0.5)

    std_normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    MPI = std_normal.cdf((mu_test - current_max - explore).div(std_test + 1e-9))

    return MPI




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
                 acquisition=expected_improvement, normalize=False,
                 normalize_y=False, max_delta=10):

        self.acquisition = acquisition

        self.max_delta = max_delta
        self.y_mean = torch.tensor(0.)
        self.y_std = torch.tensor(1.)

        self.normalize = normalize
        self.normalize_y = normalize_y
        self.kernel = kernel

        if train_x is not None:
            self.train_x = train_x.float().clone()
            self.train_y = train_y.float().clone()

            if self.normalize:
                self.train_x = self.train_x.div(self.max_delta)

        else:
            self.train_x = None
            self.train_y = None
            # self.y_mean = self.train_y.mean()
            # self.y_std = self.train_y.std()
            # self.train_y = (self.train_y - self.y_mean).div(self.y_std)

        self.surrogate_lh = gpytorch.likelihoods.GaussianLikelihood()
        # self.surrogate_lh.noise.data[0] = -1.
        self.surrogate = Surrogate(self.train_x, self.train_y, self.surrogate_lh,
                                   kernel=self.kernel)

    def train_surrogate(self, iters=50, lr=0.01, overwrite=False):
        if overwrite:
            ## I think it might be useful to wipe out the GP each time #
            self.surrogate_lh = gpytorch.likelihoods.GaussianLikelihood()
            self.surrogate_lh.noise.data[0] = -2.
            self.surrogate = Surrogate(self.train_x, self.train_y, self.surrogate_lh,
                                       kernel=self.kernel)


        if self.train_x is not None:

            # print("train x = ", self.train_x)
            # print("train y = ", self.train_y)
            # print(self.surrogate)
            # self.surrogate_lh.noise.data = torch.tensor(-2.)
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
                # print(loss.item())

    def acquire(self, **kwargs):
        """
        This will run the acquisition for the bayesian optimization method

        Right now it's fixed to just pick the best integer in [0, 1000],
        but further down the line we should figure out a way to get a viable
        range for CWND sizes.
        """
        jitter_num = 5 # maximum points to jitter from boundary if hit
        test_size = 200 # how many bins to break test domain into


        test_points = torch.linspace(-self.max_delta, self.max_delta, test_size).float()

        if self.normalize:
            int_test_points = test_points.clone()
            test_points = test_points.div(self.max_delta)

        if self.train_x is None:
            ## if we haven't passed in training data
            ## then just pick a random point and return
            if self.normalize:
                return np.random.choice(int_test_points)
            else:
                return np.random.choice(test_points)

        # print("low test = ", test_points.min())
        # print("high test = ", test_points.max())
        self.surrogate.eval()
        self.surrogate_lh.eval()

        # test_dist = self.surrogate_lh(self.surrogate(test_points))
        # best_y, ind = self.train_y.max(0)

        acquisition = self.acquisition(self, test_points, **kwargs)
        best_ac, ind = acquisition.max(0)
        if ind == test_points.numel()-1:
            # print("hitting boundary")
            ind -= np.random.choice(jitter_num)
        elif ind == 0:
            # print("hitting boundary")
            jit = np.random.choice(jitter_num)
            ind = jit
        if self.normalize:
            # print(int_test_points)
            rtrn_val = int_test_points[ind]
            # print("returning ", rtrn_val)
            return rtrn_val
        else:
            return test_points[ind]

    def update_obs(self, x, y, max_obs=None):
        '''
        Step window controls the maximum number of observations allowed
        '''
        # print("before update: ")
        # print(self.surrogate.train_inputs[0].shape)
        if self.train_x is None:
            self.train_x = x
            self.train_y = y
            self.y_mean = train_y.mean()
            self.y_std = torch.tensor(1.)

        else:
            ## if we're normalizing then de-normalize the previous observations
            if self.normalize:
                self.train_x = self.train_x * self.max_delta
            if self.normalize_y:
                self.train_y = self.train_y * self.y_std + self.y_mean

            ## now concatenate everything together
            self.train_x = torch.cat((self.train_x, x))
            self.train_y = torch.cat((self.train_y, y))

        if max_obs is not None and self.train_x.numel() > max_obs:
            ## cutoff previous observations if needed ##
            self.train_x = self.train_x[-max_obs:]
            self.train_y = self.train_y[-max_obs:]

        ## at this point everything _should_ be unnormalized, so fix that ##
        if self.normalize:
            self.train_x = self.train_x.div(self.max_delta)
        if self.normalize_y:
            self.y_mean = self.train_y.mean()
            if self.train_y.numel() > 1:
                self.y_std = self.train_y.std()
            else:
                self.y_std = torch.tensor(1.)

            self.train_y = (self.train_y - self.y_mean).div(self.y_std)

        self.surrogate.set_train_data(self.train_x, self.train_y, strict=False)
        # print("after update: ")
        # print(self.surrogate.train_inputs[0].shape)

    def get_pred_dist(self, test_x, compute_cov=False):
        self.surrogate.eval()
        self.surrogate_lh.eval()

        pred = self.surrogate(test_x)
        if compute_cov:
            pred = self.surrogate_lh(pred)

        return pred
