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

    max_action = 2.
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

        rates = torch.rand(n_sender, n_start, 1) * 2 * max_action - max_action
        rwrds = torch.zeros(n_sender, n_start)
        for rind in range(n_start):
            rwrds[:, rind] = torch.tensor(env.step(rates[:, rind, :])[1])

        BayesOpter = []
        for sndr in range(n_sender):
            BayesOpter.append(BayesOpt(rates[sndr, :, 0], rwrds[sndr, :],
                                      normalize=False, max_delta=max_action,
                                      acquisition=MPI))

        for ii in range(rnds):

            ## TRAIN MODELS AND ACQUIRE ##
            next_rates = torch.zeros(n_sender, 1)
            for sndr, bo in enumerate(BayesOpter):
                bo.train_surrogate(iters=250, overwrite=True)
                next_rates[sndr, 0] = bo.acquire(explore=0.1).item()

            ## save ##
            saved_rates[tt, ii, :, :] = next_rates

            ## run the network ##
            net_out = env.step(next_rates)
            rwrds = torch.tensor(env.step(next_rates)[1])
            saved_rwrds[tt, ii, :] = torch.tensor(rwrds)

            ## update observations
            for sndr, bo in enumerate(BayesOpter):
                new_x = next_rates[sndr, :].float().clone()
                new_y = rwrds[sndr].unsqueeze(-1).float().clone()

                bo.update_obs(new_x, new_y, max_obs=max_obs)

            for sndr in range(n_sender):
                save_thruput[tt, ii, sndr] = env.senders[sndr].get_run_data().get("recv rate")
                save_loss[tt, ii, sndr] = env.senders[sndr].get_run_data().get("loss ratio")
                save_latency[tt, ii, sndr] = env.senders[sndr].get_run_data().get("avg latency")
            save_events[tt, ii] = net_out[2]

        print("trial", tt, " done")

        torch.save(saved_rates, fpath + "separate_rates.pt")
        torch.save(saved_rwrds, fpath + "separate_rwrds.pt")
        torch.save(save_loss, fpath + "separate_loss.pt")
        torch.save(save_thruput, fpath + "separate_thruput.pt")
        torch.save(save_latency, fpath + "separate_latency.pt")
        torch.save(save_events, fpath + "separate_events.pt")


if __name__ == '__main__':
    main()
