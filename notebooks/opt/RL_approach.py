import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import datetime
import pandas as pd
import sys
import os
import math
sys.path.append('../..')
import scipy.optimize as opt
from models.optim.SEIR_Discrete import SEIR_Discrete



class Environment:
    def __init__(self, graph):
        self.simulator = SEIR_Env

    def random_action(self):
        action = random.sample(self.simulator.possible_nodes,int(min(self.simulator.n,self.simulator.budget)))
        return action

    def policy(self, state, observation, eps=0.1):
        if np.random.rand() < eps:
            return self.random_action()
        else:
            return self.greedy_action(state,observation) ###   

    def calc_pseudo_reward(self,state):
        reward=self.simulator.n-sum(state)
        return reward

    def run_episode(self, eps=0.1, discount=0.98):
        S, A, R = [], [], [], []
        cumulative_reward = 0
        self.simulator.reset()
        observation = self.simulator.observe()
        state = 0.5 * np.ones(self.simulator.n)
        O.append(observation)
        for t in range(self.simulator.T):
            state = self.belief_state(observation, state)
            S.append(state)
            action = self.policy(state, observation, eps)
            self.simulator.perform(active_screen_list=action)#Transition Happen
            state_ = self.belief_transition(observation, action, state) #S'
            reward = self.calc_pseudo_reward(state)
            observation = self.simulator.observe()
            state=state_
            A.append(action)
            R.append(reward)
            S.append(state_)
            cumulative_reward += reward * (discount**t)
        return S, A, R, cumulative_reward

    
    def run_episode_real(self, eps=0.1, discount=0.98):
        O, S, A, R = [], [], [], []
        cumulative_reward = 0
        self.simulator.reset()
        observation = self.simulator.observe()
        state = 0.5 * np.ones(self.simulator.n)
        O.append(observation)
        for t in range(self.simulator.T):
            state = self.belief_state(observation, state)
            S.append(state)
            action = self.policy(state, observation, eps)
            self.simulator.perform(active_screen_list=action)#Transition Happen
            reward = self.simulator.calc_reward()
            state = self.belief_transition(observation, action, state) #S'
            observation = self.simulator.observe()
            A.append(action)
            R.append(reward)
            O.append(observation)
            S.append(state)
            cumulative_reward += reward * (discount**t)
        return O, S, A, R, cumulative_reward

if __name__ == '__main__':
    method='FQI'

    if method == 'DQN':
        model = DeepQNetworks(graph=g)
    elif method == 'FQI':
        model = FittedQIteration(graph=g)
    else:
        raise ValueError('Not implemented method')
    discount=0.9
    # random policy
    cumulative_rewards = []
    for i in range(10):
        O, S, A, R, cumulative_reward = model.run_episode_real(eps=1, discount=discount)
        cumulative_rewards.append(cumulative_reward)
    print('random action reward:', np.mean(cumulative_rewards))
    eps=0.05
    if method == 'DQN':
        model.fit(discount=discount, num_iterations=100, eps= eps, num_epochs=100) # for DQN
    elif method == 'FQI':
        model.fit(num_refits=10, num_episodes=10,eps=eps, discount=discount) # for FQI
    else:
        raise ValueError('Not implemented method')

    # optimal policy
    cumulative_rewards = []
    for i in range(10):
        O, S, A, R, cumulative_reward = model.run_episode_real(eps=0, discount=discount)
        cumulative_rewards.append(cumulative_reward)
    print('optimal reward:', np.mean(cumulative_rewards))