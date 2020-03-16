import math
import torch
import ntwrk
from ntwrk.bayesopt import BayesOpt, expected_improvement, max_mean

import gym
import numpy as np


class BayesOptCongestion(object):
    """
    This is the class that holds the bayesian optimization routine
    and the network environment that will allow simulations to be run from a
    high level with just a few commands
    """

    def __init__(self, bayesopt, env, step_window=None):
        self.bayesopt = bayesopt # the bayes opt class
        self.env = env # the network env
        self.step_window = step_window # the number of previous steps to use to compute next action

        if self.bayesopt.train_x is None:
            ## if bayesopt is initialized with no observations then generate one ##
            self.init_rwrd()

    def init_rwrd(self):
        ## generate an observation so that bayesopt can be run ##
        # train_x = torch.tensor(self.env.senders[0].starting_rate).unsqueeze(0)
        train_x = torch.rand(1)
        train_y = self.step(train_x)

        self.bayesopt.update_obs(train_x, train_y)

    def step(self, action):
        # return torch.tensor(self.env.step(action)[1]).unsqueeze(0)
        return self.env(action)

    def run_bayesopt(self, iters=10, train_iters=200, explore=1.):
        rates = self.bayesopt.train_x
        rwrds = self.bayesopt.train_y

        for iter in range(iters):
            # train and acquire #
            self.bayesopt.train_surrogate(train_iters, overwrite=False)
            # next_rate = self.bayesopt.acquire(explore=explore).unsqueeze(0)
            next_rate = self.bayesopt.acquire(explore=explore).unsqueeze(-1)
            rwrd = self.step(next_rate)

            # print(rates)
            # print(next_rate)
            # rates = torch.cat((rates, next_rate), 0)
            # rwrds = torch.cat((rwrds, rwrd), 0)

            rates = torch.cat((rates, next_rate))
            rwrds = torch.cat((rwrds, rwrd))

            # update obs #
            self.bayesopt.update_obs(next_rate, rwrd, max_obs=self.step_window)

            print("train x = ", self.bayesopt.train_x)
            print("train y = ", self.bayesopt.train_y)


        return rates, rwrds
