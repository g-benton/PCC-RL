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
    max_record = torch.arange(1, 10)
    n_jumps = max_record.numel()
    n_iters = 100
    n_trials = 20
    max_x = 1000
    n_start = 3
    max_jump = 500

    save_rewards = torch.zeros(n_trials, n_jumps, n_iters)
    save_rates = torch.zeros(n_trials, n_jumps, n_iters)

    ## BIG LOOP FOR EACH MAX JUMP ##
    for tt in range(n_trials):
        for rcrd_ind, rcrd in enumerate(max_record):
            print("running record length ", rcrd)
            ## set up env and start with some obs ##

            env = gym.make("PccNs-v0")
            env.reset()
            rates = torch.rand(n_start)
            rwrds = torch.zeros(n_start)
            for rind, rr in enumerate(rates):
                rwrds[rind] = env.step(rr.unsqueeze(0).mul(max_x))[1].item()

            bo = BayesOpt(rates.mul(max_x), rwrds, normalize=True, max_x=max_x,
                          acquisition=expected_improvement, max_jump=max_jump)

            for ii in range(n_iters):
                try:
                    bo.train_surrogate(iters=250, overwrite=True)

                    next_rate = bo.acquire(explore=0.1).unsqueeze(0)
                    rwrd = torch.tensor(env.step(next_rate.mul(bo.max_x))[1]).unsqueeze(0)

                    save_rewards[tt, rcrd_ind, ii] = rwrd.item()
                    save_rates[tt, rcrd_ind, ii] = next_rate.item()

                    bo.update_obs(next_rate, rwrd, max_obs=rcrd)
                except:
                    save_rewards[tt, rcrd_ind, ii] = rwrd.item()
                    save_rates[tt, rcrd_ind, ii] = next_rate.item()

                # print(save_rewards[tt, rcrd_ind, ii])
                # print(save_rates[tt, rcrd_ind, ii])
                # print("\n")

        print("saving trial", tt)
        torch.save(save_rates, "saved_rates.pt")
        torch.save(save_rewards, "saved_rwrds.pt")

if __name__ == '__main__':
    main()
