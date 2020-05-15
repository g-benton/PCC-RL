import math
import torch
import ntwrk
from ntwrk.bayesopt import BayesOpt, expected_improvement, max_mean, MPI

import gym
import sys
sys.path.append("../../ntwrk/gym/")
import multi_sender_sim
import numpy as np


def main():
    n_sender = 2
    n_trials = 20

    max_action = 1.
    max_obs = 3
    n_start = 3

    rnds = 100
    saved_rates = torch.zeros(n_trials, rnds, n_sender, 1)
    saved_rwrds = torch.zeros(n_trials, rnds, n_sender)
    save_loss = torch.zeros(n_trials, rnds, n_sender)
    save_latency = torch.zeros(n_trials, rnds, n_sender)
    save_thruput = torch.zeros(n_trials, rnds, n_sender)

    save_events = torch.zeros(n_trials, rnds)

    fpath = "./saved-outputs/"

    for tt in range(n_trials):

        env = gym.make("Sndr-v0")
        env.reset(n_sender)
        deltas = torch.rand(n_start)
        rwrds = torch.zeros(n_start)
        for rnd in range(n_start):
            curr_dlta = deltas[rnd].expand(n_sender, 1)
            net_out = env.step(curr_dlta)
            rwrds[rnd] = torch.tensor(net_out[1]).sum()

        bo = BayesOpt(deltas, rwrds, normalize=True, max_delta=max_action,
                      acquisition=MPI)

        ## standard BO loop
        for ii in range(rnds):
            bo.train_surrogate(iters=250, overwrite=True)
            ac_delta = bo.acquire(explore=0.1).unsqueeze(0)

            next_rate = ac_delta.expand(n_sender, 1)
            net_out = env.step(next_rate)
            rwrds = torch.tensor(net_out[1])

            saved_rates[tt, ii] = ac_delta.item()
            saved_rwrds[tt, ii, :] = rwrds

            for sndr in range(n_sender):
                save_thruput[tt, ii, sndr] = env.senders[sndr].get_run_data().get("recv rate")
                save_loss[tt, ii, sndr] = env.senders[sndr].get_run_data().get("loss ratio")
                save_latency[tt, ii, sndr] = env.senders[sndr].get_run_data().get("avg latency")
            save_events[tt, ii] = net_out[2]
            # print(net_out)


            bo.update_obs(ac_delta, rwrds.sum().unsqueeze(-1).float(), max_obs=max_obs)

        print("trial", tt, " done")

        torch.save(saved_rates, fpath + "together_saved_rates.pt")
        torch.save(saved_rwrds, fpath + "together_saved_rwrds.pt")
        torch.save(save_loss, fpath + "together_loss.pt")
        torch.save(save_thruput, fpath + "together_thruput.pt")
        torch.save(save_latency, fpath + "together_latency.pt")
        torch.save(save_events, fpath + "together_events.pt")


if __name__ == '__main__':
    main()
