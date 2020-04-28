import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../..')
from models.optim.SEIR_Discrete_2 import SEIR_Discrete
# from SEIR_Discrete import SEIR_Discrete
from sklearn.ensemble import ExtraTreesRegressor
import pickle
import time
from tqdm import tqdm, trange

def is_fitted(sklearn_regressor):
    """Helper function to determine if a regression model from scikit-learn has
    ever been `fit`"""
    return hasattr(sklearn_regressor, 'n_outputs_')

"""Helper class to run Fitted Q Iteration on the SEIR_Discrete task."""
class FittedQIteration(object):
    def __init__(self, regressor=None):
        """Initialize simulator and regressor. Can optionally pass a custom
        `regressor` model (which must implement `fit` and `predict` -- you can
        use this to try different models like linear regression or NNs)"""
        self.simulator = SEIR_Discrete()
        self.regressor = regressor or ExtraTreesRegressor()

    def Q(self, states):
        """Return the Q function estimate of `states` for each action"""
        if not is_fitted(self.regressor):
            # If not fitted, return 0s in the right shape
            return np.zeros((len(states), self.simulator.num_actions))
        else:
            # Otherwise use the trained regression model
            return np.array([self.regressor.predict(self.encode(states, action))
                             for action in range(self.simulator.num_actions)]).T

    def policy(self, state, eps=0.1):
        """Return the epsilon-greedy action based on the current policy (or a
        random action if the Q function hasn't yet been estimated."""
        if np.random.rand() < eps or not is_fitted(self.regressor):
            return np.random.choice(self.simulator.num_actions)
        else:
            return self.Q([state]).argmax()

    def run_episode(self, eps=0.1, episode_length=200):
        """Run a single episode on the HIV simulator using the current policy.
        Can pass a custom `eps` to test out varying levels of randomness.
        Return the states visited, actions taken, and rewards received."""
        S = np.zeros((episode_length+1, self.simulator.num_states))
        A = np.zeros(episode_length)
        R = np.zeros(episode_length)

        self.simulator.reset()
        S[0] = self.simulator.STATE

        for t in range(episode_length):
            A[t] = self.policy(S[t], eps=eps)
            R[t], S[t+1] = self.simulator.perform(A[t])
        return S, A, R

    def fit_Q(self, episodes, num_iters=100, discount=0.98,episode_length=200):
        """Fit and re-fit the Q function using historical data for the
        specified number of `iters` at the specified `discount` factor"""
        S1 = np.vstack([ep[0][:-1] for ep in episodes])
        S2 = np.vstack([ep[0][1:] for ep in episodes])
        A = np.hstack([ep[1] for ep in episodes])
        R = np.hstack([ep[2] for ep in episodes])
        inputs = self.encode(S1, A)
        progress = tqdm(range(num_iters), file=sys.stdout,desc='num_iters')
        for _ in range(num_iters):
            progress.update(1)
            targets = R + discount * self.Q(S2).max(axis=1)
            self.regressor.fit(inputs, targets)
        progress.close()
    def fit(self, num_refits=10, num_episodes=15, episode_length=200, save=False):
        """Perform fitted-Q iteration. For `outer_iters` steps, gain
        `num_episodes` episodes worth of experience using the current policy
        (which is initially random), then fit or re-fit the Q-function. Return
        the full set of episode data."""
        episodes = []
        progress = tqdm(range(num_refits), file=sys.stdout,desc='num_refits')
        for i in range(num_refits):
            progress.update(1)
            for _ in range(num_episodes):
                episodes.append(self.run_episode(episode_length=episode_length))
            self.fit_Q(episodes=episodes, episode_length=episode_length)
            if save:
                with open('./fqi-regressor-iter-{}.pkl'.format(i+1), 'wb') as f:
                    pickle.dump(self.regressor, f)
        progress.close()
        return episodes

    def encode(self, states, actions):
        """Encode states and actions as a single input array, suitable for
        passing into the `fit` method of a scikit-learn model."""
        if isinstance(actions, int):
            actions = [actions] * len(states)
        actions = np.array([[a] for a in actions])
        states = np.array(states)
        return np.hstack([states, actions])

if __name__ == '__main__':
    print('Here goes nothing')
    
    num_refits=10
    num_episodes=15
    episode_length=100
    First_time=False
    if First_time:
        env=FittedQIteration()
        episodes=env.fit( num_refits=num_refits, num_episodes=num_episodes,episode_length=episode_length)
     
        with open('Result_refits={}_episodes={}_episode_length={}.pickle'.format(num_refits,num_episodes,episode_length), 'wb') as f:
                    pickle.dump([env,episodes], f)
    else:            
        with open('Result_refits={}_episodes={}_episode_length={}.pickle'.format(num_refits,num_episodes,episode_length), 'rb') as f:
            X = pickle.load(f)  
        env=X[0]
        episodes=X[1]
    
    discount=0.98
    real_episodes=env.run_episode(eps=0,episode_length=episode_length)
    S=real_episodes[0]
    A=real_episodes[1]
    R=real_episodes[2]
    best_r=0
    for i in range(len(R)):
        best_r+=R[i]*(discount**i)
    

    random_action_episodes=env.run_episode(eps=1,episode_length=episode_length)
    S_r=random_action_episodes[0]
    A_r=random_action_episodes[1]
    R_r=random_action_episodes[2]
    random_r=0
    for i in range(len(R_r)):
        random_r+=R_r[i]*(discount**i)
    test_r=0    
    
    state=real_episodes[0][0]
    
    estimate_r=env.Q([state])[0]
    print("Estimate Reward(Discounted):")
    print(max(estimate_r))
    print("Real Reward(Discounted):")
    print(best_r)
    print("Random Action Reward(Discounted):")
    print(random_r)
    # S_=[]
    # E=[]
    # I=[]
    # for i in range(len(S)):
    #     S_.append(S[i][0])
    #     E.append(S[i][1])
    #     I.append(S[i][2])
    # plt.plot(range(len(S_)),S_)
    # plt.plot(range(len(E)),E)
    # plt.plot(range(len(I)),I)
    S_=[]
    E=[]
    I=[]
    for i in range(len(S_r)):
        S_.append(S_r[i][0])
        E.append(S_r[i][1])
        I.append(S_r[i][2])
    plt.plot(range(len(S_)),S_)
    plt.plot(range(len(E)),E)
    plt.plot(range(len(I)),I)
    