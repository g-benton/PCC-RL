import math
import torch
import ntwrk
from ntwrk.bayesopt import BayesOpt, expected_improvement, max_mean

import gym
import sys
sys.path.append("../../ntwrk/gym/")
import network_sim
import numpy as np
import matplotlib.pyplot as plt

def main():
    torch.random.manual_seed(88)

    ## SETUP AND DEFS ##
    max_jumps = torch.arange(100, 1001, step=100)
    n_jumps = max_jumps.numel()
    n_iters = 100
    max_x = 1000
    n_start = 3

    save_rewards = torch.zeros(n_jumps, n_iters)
    save_rates = torch.zeros(n_jumps, n_iters)

    ## BIG LOOP FOR EACH MAX JUMP ##
    for jump_ind, jump in enumerate(max_jumps):
        print("running for jump size ", jump)
        ## set up env and start with some obs ##

        env = gym.make("PccNs-v0")
        env.reset()
        rates = torch.rand(n_start)
        rwrds = torch.zeros(n_start)
        for rind, rr in enumerate(rates):
            rwrds[rind] = env.step(rr.unsqueeze(0).mul(max_x))[1].item()

        bo = BayesOpt(rates.mul(max_x), rwrds, normalize=True, max_x=max_x,
                      acquistion=expected_improvement, max_jump=jump)

        for ii in range(n_iters):
            bo.train_surrogate(iters=250, overwrite=True)
            next_rate = bo.acquire(explore=0.1).unsqueeze(0)
            rwrd = torch.tensor(env.step(next_rate.mul(bo.max_x))[1]).unsqueeze(0)

            save_rewards[jump_ind, ii] = rwrd.item()
            save_rates[jump_ind, ii] = next_rate.item()

            bo.update_obs(next_rate, rwrd, max_obs=5)


        print("saving jump size = ", jump)
        torch.save(save_rates, "max_jump_rates.pt")
        torch.save(save_rewards, "max_jump_rwrds.pt")

if __name__ == '__main__':
    main()
