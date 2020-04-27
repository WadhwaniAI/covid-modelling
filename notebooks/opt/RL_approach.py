import sys
import os
import math
import numpy as np
# sys.path.append('../..')
# from models.optim.SEIR_Discrete import SEIR_Discrete
from SEIR_Discrete import SEIR_Discrete
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

    def fit_Q(self, episodes, num_iters=100, discount=0.98):
        """Fit and re-fit the Q function using historical data for the
        specified number of `iters` at the specified `discount` factor"""
        S1 = np.vstack([ep[0][:-1] for ep in episodes])
        S2 = np.vstack([ep[0][1:] for ep in episodes])
        A = np.hstack([ep[1] for ep in episodes])
        R = np.hstack([ep[2] for ep in episodes])
        inputs = self.encode(S1, A)
        progress = tqdm(range(num_iters), file=sys.stdout,desc='num_iters')
        for _ in range(num_iters):
            sys.stdout.flush()
            progress.update()
            progress.refresh()
            targets = R + discount * self.Q(S2).max(axis=1)
            self.regressor.fit(inputs, targets)

    def fit(self, num_refits=10, num_episodes=15, save=False):
        """Perform fitted-Q iteration. For `outer_iters` steps, gain
        `num_episodes` episodes worth of experience using the current policy
        (which is initially random), then fit or re-fit the Q-function. Return
        the full set of episode data."""
        episodes = []
        progress = tqdm(range(num_refits), file=sys.stdout,desc='num_refits')
        for i in range(num_refits):
            progress.update(1)
            for _ in range(num_episodes):
                episodes.append(self.run_episode())
            self.fit_Q(episodes)

            if save:
                with open('./fqi-regressor-iter-{}.pkl'.format(i+1), 'wb') as f:
                    pickle.dump(self.regressor, f)

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
    env=FittedQIteration()
    episodes=env.fit( num_refits=10, num_episodes=30)
 
    with open('Result_30EP.pickle', 'wb') as f:
                pickle.dump([env,episodes], f)
                
                
                
    # with open('Result_30EP.pickle', 'rb') as f:
    #     X = pickle.load(f)  
    # env=X[0]
    # episodes=X[1]
    
###################D1##############################
    
    discount=0.98
    real_episodes=env.run_episode(eps=0)
    reward=real_episodes[1]
    test_r=0
    
    
    state_t=real_episodes[0][0]
    state=state_t[2:]
    
    estimate_r=env.Q([state])[0]
    print(state_t[:2])
    print(estimate_r[2])
    
    
    for i in range(len(reward)):
        test_r+=reward[i]*(discount**i)
    all_r=[]
    for i in range(10): 
        e_r=0
        for j in range(30): 
#            ep=episodes[i*10+j]
            ep=episodes[i*10+j]
            reward=ep[1]
            r=0
            for e in range(len(reward)):
                r+=reward[e]*(discount**e)
            e_r+=r
        all_r.append(e_r/30)
#        
    all_f=[]
    feature_num=1-1
    f_t=np.zeros((200,150))
    for i in range(10): 
        for j in range(15):
            ep=episodes[i*10+j]
            feature=ep[0]
            for e in range(len(reward)):
                f_t[e][(15*i+j)]=feature[e][feature_num+2]

    ff_t=np.zeros(200)
    for i in range(1): 
        for j in range(1):
            ep=real_episodes
            feature=ep[0]
            for e in range(len(reward)):
                
                ff_t[e]=feature[e][feature_num+2]    

###################D2##############################
#    Eps=1
#    feature_num=6-1
#    
#    real_episodes=[]
##    for i in range(100):
##        real_episodes.append(env.run_episode(eps=Eps))
###        if(i>=50):
###            real_episodes.append(env.run_episode(eps=Eps))
##    env2=FittedQIteration()
##    env2.fit_Q(real_episodes)
##    with open('hw2Result_encode_'+str(Eps)+'.pickle', 'wb') as f:
##            pickle.dump(env2, f)
##            
#            
#    with open('hw2Result_encode_'+str(Eps)+'.pickle', 'rb') as f:
#        env2 = pickle.load(f)
#    
#    real_episodes2=env2.run_episode(eps=0)
#    discount=0.98
#    reward=real_episodes2[1]
#    state_t=real_episodes2[0][0]
#    state=state_t[2:]
#    test_r=0
#    estimate_r=env2.Q([state])[0]
#    print(state_t[:2])
#    print(estimate_r[2])
#    for i in range(len(reward)):
#        test_r+=reward[i]*(discount**i)
#    print(test_r)
#    all_r=[]
#    for i in range(10): 
#        e_r=0
#        for j in range(15): 
#            ep=episodes[i*10+j]
#            reward=ep[1]
#            r=0
#            for e in range(len(reward)):
#                r+=reward[e]*(discount**e)
#            e_r+=r
#        all_r.append(e_r/15)
##        
#    all_f=[]
#    
#    f_t=np.zeros((200,150))
#    for i in range(10): 
#        for j in range(15):
#            ep=episodes[i*10+j]
#            feature=ep[0]
#            for e in range(len(reward)):
#                f_t[e][(15*i+j)]=feature[e][feature_num+2]
#
#    ff_t=np.zeros(200)
#    for i in range(1): 
#        for j in range(1):
#            ep=real_episodes2
#            feature=ep[0]
#            for e in range(len(reward)):
#                
#                ff_t[e]=feature[e][feature_num+2]    