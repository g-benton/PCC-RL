import math
import torch
import ntwrk
from ntwrk.bayesopt import BayesOpt, expected_improvement, max_mean

import gym
import sys
sys.path.append("../../ntwrk/gym/")
import multi_sender_sim
import numpy as np


def main():
    n_sender = 2
    n_trials = 20

    max_action = 2.
    max_obs = 3
    n_start = 3

    rnds = 1000
    saved_rates = torch.zeros(n_trials, rnds, n_sender, 1)
    saved_rwrds = torch.zeros(n_trials, rnds, n_sender)

    fpath = "./saved-outputs/"

    for tt in range(n_trials):

        env = gym.make("Sndr-v0")
        env.reset(n_sender)
        deltas = torch.rand(n_start)
        rwrds = torch.zeros(n_start)
        for rnd in range(n_start):
            curr_dlta = deltas[rnd].expand(n_sender, 1)
            rwrds[rnd] = torch.tensor(env.step(curr_dlta)[1]).sum()

        bo = BayesOpt(deltas, rwrds, normalize=True, max_delta=max_action,
                      acquisition=expected_improvement)

        ## standard BO loop
        for ii in range(rnds):
            bo.train_surrogate(iters=500, overwrite=True)
            ac_delta = bo.acquire(explore=0.1).unsqueeze(0)

            next_rate = ac_delta.expand(n_sender, 1)
            rwrds = torch.tensor(env.step(next_rate)[1])

            saved_rates[tt, ii] = ac_delta.item()
            saved_rwrds[tt, ii, :] = rwrds

            bo.update_obs(ac_delta, rwrds.sum().unsqueeze(-1).float(), max_obs=max_obs)

        print("trial", tt, " done")

        torch.save(saved_rates, fpath + "together_saved_rates.pt")
        torch.save(saved_rwrds, fpath + "together_saved_rwrds.pt")


if __name__ == '__main__':
    main()
