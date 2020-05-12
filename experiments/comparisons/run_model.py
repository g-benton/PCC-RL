import math
import torch
import os
import ntwrk
from ntwrk.bayesopt import BayesOpt, expected_improvement, max_mean

import gym
import sys
sys.path.append("../ntwrk/gym/")
import network_sim
import numpy as np
import matplotlib.pyplot as plt

MAX_RATE = 1000
MIN_RATE = 40
DELTA_SCALE = 0.025

env = gym.make("PccNs-v0")
env.reset()

max_x = 1000.
n_start = 3
rates = torch.rand(n_start)
rwrds = torch.zeros(n_start)
for rind, rr in enumerate(rates):
    rwrds[rind] = env.step(rr.unsqueeze(0).mul(max_x))[1].item()

bo = BayesOpt(rates.mul(max_x), rwrds, normalize=True, max_x=max_x, acquisition=expected_improvement,
             max_jump=300)

# bayesOpt
rnds = 1000
saved_rwrds = torch.zeros(rnds)
bayes_rates = torch.zeros(rnds)
test_points = torch.arange(1, 1000).float().div(max_x)
for ii in range(rnds):
    bo.train_surrogate(iters=500, overwrite=True)
    next_rate = bo.acquire(explore=0.1).unsqueeze(0)
    bayes_rates[ii] = next_rate
    # print(next_rate.shape)
    # print("next rate = ", next_rate)
    output = env.step(next_rate.mul(bo.max_x))
    rwrd = torch.tensor(output[1]).unsqueeze(0)
    saved_rwrds[ii] = rwrd.item()
    bo.update_obs(next_rate, rwrd, max_obs=4)

np.savez('bayes_model_out', bayes_rates=bayes_rates, saved_rwrds=saved_rwrds)

# def apply_rate_delta(rate, delta):
# 	delta *= DELTA_SCALE
# 	if delta >= 0.0:
# 		send_rate = set_rate(rate * (1.0 + delta))
# 	else:
# 		send_rate = set_rate(rate / (1.0 - delta))
# 	return send_rate

# def set_rate(rate):
#     if rate > MAX_RATE:
#         rate = MAX_RATE
#     if rate < MIN_RATE:
#         rate = MIN_RATE

#     return rate

# # pcc-rl
# rnds = 1000
# pcc_rwrds = torch.zeros(rnds)
# pcc_rates = torch.zeros(rnds)
# actions = torch.zeros(rnds, 2)
# #send_rate = rates[-1]
# send_rate = MIN_RATE + torch.rand(1) * (MAX_RATE - MIN_RATE)
# for ii in range(rnds):
# 	pcc_rates[ii] = send_rate
# 	output = env.step(send_rate)
# 	obs = output[0]
# 	pcc_rwrds[ii] = obs[1]
# 	obs = obs.reshape(1, 30)
# 	np.save('obs_data', obs)
# 	os.system('saved_model_cli run --dir pcc_saved_models/model_B --tag_set "serve" --signature_def "serving_default" --inputs "ob=obs_data.npy" --outdir output_act')
# 	data1 = np.load('output_act/act.npy')
# 	data2 = np.load('output_act/stochastic_act.npy')
# 	action = torch.tensor([data1[0][0], data2[0][0]])
# 	os.system('rm -rf output_act/*')
# 	actions[ii,:] = action
# 	print("Action value %f: " % action[0])
# 	send_rate = apply_rate_delta(send_rate, action[0])

# np.savez('run_model_out', pcc_rates=pcc_rates, pcc_rwrds=pcc_rwrds, actions=actions)