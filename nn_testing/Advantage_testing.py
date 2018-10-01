import numpy as np
# import lasagne
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import math
import random
import json
# import scipy
import atexit, numpy as np, scipy, sys, os.path as osp
from scipy import signal, misc
from collections import defaultdict

from model.ModelUtil import *


def sum_furture_discounted_rewards(rewards, discount_factor):
    discounts=[]
    for k in range(len(rewards)):
        discounts.append(0)
        for i in range(len(rewards)-k):
            discounts[k] += rewards[k+i] * math.pow(discount_factor, i)
            # print ("discounts: ", k, " ", i, " reward:", rewards[k+i],  " discounts", discounts)
    return discounts
"""
def compute_advantage(discounted_rewards, rewards, discount_factor):
    discounts = []
    for i in range(len(discounted_rewards)-1):
        discounts.append(((discounted_rewards[i+1] * discount_factor) + rewards[i+1]) - discounted_rewards[i])
    return discounts
"""
def compute_advantage2(b1, rewards, discount_factor):
    discounts = deltas = rewards + discount_factor*b1[1:] - b1[:-1]
    return discounts

if __name__ == "__main__":
    
    discount_factor = 0.9
    if (len(sys.argv) == 2):
        discount_factor = float(sys.argv[1])
    
    print ("Discount Factor: ", discount_factor)
    # rewards=[0.2, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    # rewards=[0.2, 0.25, 0.2, 0.2, 0.2, 0.7]
    rewards=[0.082, 0.14, 0.2332, 0.2025, 0.117, 0.08025085, 0.08025085]
    # rewards2 = rewards[1:]
    print ("Rewards: ", rewards)
    discounts = sum_furture_discounted_rewards(rewards, discount_factor)
    print("Discounts: ", discounts)
    dis = discounted_rewards(np.array(rewards), discount_factor)
    print("Discounts: ", dis)
    advantage = compute_advantage(discounts, rewards, discount_factor)
    print ("Advantage: ", advantage)
    discounts.append(0.0)
    dis = np.append(dis, 0.0)
    adv = compute_advantage2(np.array(dis), np.array(rewards), discount_factor)
    print ("Advantage: ", adv)
    
    