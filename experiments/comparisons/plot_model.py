import math
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

data_pcc = np.load('run_model_out.npz')

data_bayes = np.load('bayes_model_out.npz')

iterations = np.arange(1,1001)
plt.title("PCC-RL vs BOCC send rates") 
plt.xlabel("Iteration") 
plt.ylabel("Send rate") 
plt.plot(iterations, data_pcc['pcc_rates'], 'r', label='PCC-RL') 
plt.plot(iterations, data_bayes['bayes_rates'], 'b', label='BOCC')
plt.ylim((0,1200))
plt.legend(loc="upper right")
plt.savefig('bayes_vs_pcc_rate.pdf')
plt.show()

plt.title("PCC-RL vs BOCC rewards") 
plt.xlabel("Iteration") 
plt.ylabel("Reward") 
plt.plot(iterations, data_pcc['pcc_rwrds'], 'r', label='PCC-RL') 
plt.plot(iterations, data_bayes['saved_rwrds'], 'b', label='BOCC')
plt.legend(loc="center right")
plt.savefig('bayes_vs_pcc_rwrd.pdf')
plt.show()