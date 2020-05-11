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
    max_deltas = torch.tensor([1, 2, 5, 10, 15, 20, 25, 50, 75, 100])
    n_jumps = max_deltas.numel()
    n_iters = 100
    n_trials = 20
    n_start = 3
    max_delta = 2.
    max_obs = 3

    save_rewards = torch.zeros(n_trials, n_iters)
    save_thruput = torch.zeros(n_trials, n_iters)
    save_loss = torch.zeros(n_trials, n_iters)
    save_latency = torch.zeros(n_trials, n_iters)
    save_actions = torch.zeros(n_trials, n_iters)

    ## BIG LOOP FOR EACH MAX JUMP ##
    for tt in range(n_trials):

        env = gym.make("PccNs-v0")
        env.reset()
        deltas = torch.rand(n_start) * 2 * max_delta - max_delta
        rwrds = torch.zeros(n_start)
        for rind, rr in enumerate(deltas):
            rwrds[rind] = env.step(rr.unsqueeze(0))[1].item()

        bo = BayesOpt(deltas, rwrds, normalize=True, max_delta=max_delta,
                      acquisition=expected_improvement)

        for ii in range(n_iters):
            bo.train_surrogate(iters=250, overwrite=True)
            next_rate = bo.acquire(explore=0.01).unsqueeze(0)
            rwrd = torch.tensor(env.step(next_rate)[1]).unsqueeze(0)

            save_rewards[tt, ii] = rwrd.item()
            save_actions[tt, ii] = next_rate.item()

            save_thruput[tt, ii] = env.senders[0].get_run_data().get("recv rate")
            save_loss[tt, ii] = env.senders[0].get_run_data().get("loss ratio")
            save_latency[tt, ii] = env.senders[0].get_run_data().get("avg latency")
            bo.update_obs(next_rate, rwrd, max_obs=max_obs)


        print("saving trial ", tt)
        torch.save(save_actions, "actions.pt")
        torch.save(save_rewards, "rwrds.pt")
        torch.save(save_loss, "loss.pt")
        torch.save(save_thruput, "thruput.pt")
        torch.save(save_latency, "latency.pt")

if __name__ == '__main__':
    main()
