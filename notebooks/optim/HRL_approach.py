import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
sys.path.append('../..')
# from models.optim.SEIR_Discrete import SEIR_Discrete
#from models.optim.sir_discrete import SIR_Discrete
from sir_discrete3 import SIR_Discrete
from sklearn.ensemble import ExtraTreesRegressor
import pickle
import time
import random
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
        # self.simulator = SEIR_Discrete()
        self.simulator = SIR_Discrete()
        self.regressor = regressor or ExtraTreesRegressor()

    def Q(self, states):
        """Return the Q function estimate of `states` for each action"""
        Q=np.zeros((len(states), self.simulator.num_actions,self.simulator.max_duration))
        if not is_fitted(self.regressor):
            # If not fitted, return 0s in the right shape
            return Q
        else:
            # Otherwise use the trained regression model
            SA_table=[]
            for i in range(len(states)):
                for j in range(self.simulator.num_actions):
                    for k in range(self.simulator.max_duration):
                        SA_table.append(self.encode([states[i]], [j, k])[0])
#                        SA_table.append([j, k])
#                        Q[i,j,k]=self.regressor.predict(self.encode([states[i]], [j, k]))
#            print(SA_table)
#            print(SA_table[6])
            Qt=self.regressor.predict(SA_table)
#            print(self.regressor.predict([[0,3]]))
#            print(SA_table)
#            print('QT')
#            print(Qt)
#            print(self.regressor.predict([SA_table[0]]))
#            print(self.regressor.predict([SA_table[6]]))
#            print(Qt)
            t=0
            for i in range(len(states)):
                for j in range(self.simulator.num_actions):
                    for k in range(self.simulator.max_duration):
                        state=SA_table[t]
                        if(state[6]):
                            Q[i,j,k]=0
                        else:
                            Q[i,j,k]=Qt[t]
                        t+=1
#            np.set_printoptions(precision=3)
#            print(Q)
            return Q
#            return np.array([self.regressor.predict(self.encode(states, [action, duration]))
#                             for action, duration in itertools.product(range(self.simulator.num_actions), range(self.simulator.max_duration))])


    def policy(self, state, eps=0.1):
        """Return the epsilon-greedy action based on the current policy (or a
        random action if the Q function hasn't yet been estimated."""
        a=np.zeros(2)
        if np.random.rand() < eps or not is_fitted(self.regressor):
            a[0]=np.random.choice(self.simulator.num_actions)
            a[1]=np.random.choice(self.simulator.max_duration)
        else:
            # Find index of maximum value from 2D numpy array
            Table=self.Q([state])[0]
#            print(Table)
            result = np.where(Table== np.amax(Table))
#            print(result)
            listOfCordinates = list(zip(result[0], result[1]))
#            print(listOfCordinates)
            # travese over the list of cordinates
            if len(listOfCordinates)>0:
                cord=random.choice(listOfCordinates)
                a[0]=cord[0]
                a[1]=cord[1]
            else:
                a=[0,0]
#            Duration=a[1]*10+10
#            if state[3]<self.simulator.get_action_cost(a[0])*Duration:
#                a[0]=0
#            if state[4]+Duration>self.simulator.T:
#                a[1]=math.floor((self.simulator.T-state[4])/10)
            
            
#            print(a)
        return a


    def run_episode(self, eps=0.1):
        """Run a single episode on the SEIR_Discrete using the current policy.
        Can pass a custom `eps` to test out varying levels of randomness.
        Return the states visited, actions taken, and rewards received."""
        S=[]
        A=[]
        R=[]
        A_S=[]
        self.simulator.reset()
        s=self.simulator.STATE
        S.append(s)
        while self.simulator.STATE[6]==False:
            a=self.policy(s, eps=eps)
#            print(a)
#            print('-----------')
            r, s_ , a_s= self.simulator.perform_leader(a[0],a[1])
            s=s_
            S.append(s)
            A.append(a)
            R.append(r)
            A_S=A_S+a_s
#        print(A)
        return S, A, R, A_S



    def fit_Q(self, episodes, num_iters=2, discount=0.9999):
        """Fit and re-fit the Q function using historical data for the
        specified number of `iters` at the specified `discount` factor"""
        S1 = np.vstack([ep[0][:-1] for ep in episodes])
        S2 = np.vstack([ep[0][1:] for ep in episodes])
        A = np.vstack([ep[1] for ep in episodes])
        R = np.hstack([ep[2] for ep in episodes])
        inputs = self.encode(S1, A)
#        print(A)
#        print(R)
#        progress = tqdm(range(num_iters), file=sys.stdout,desc='num_iters')
        for _ in range(num_iters):
#            progress.update(1)
#            targets = R + discount * self.Q(S2).max(axis=1)
            targets = R + discount * ((self.Q(S2).max(axis=2)).max(axis=1))
#            targets = R
#            print(targets)
#            alpha=1
#            targets = (self.Q(S1).max(axis=2)).max(axis=1)+alpha*(R + discount * (self.Q(S2).max(axis=2)).max(axis=1)-(self.Q(S1).max(axis=2)).max(axis=1))

            self.regressor.fit(inputs, targets)
#            self.regressor.fit(A, targets)
#        progress.close()
        
        
        
    def fit(self, num_refits=10, num_episodes=15, discount=0.9999, save=False):
        """Perform fitted-Q iteration. For `outer_iters` steps, gain
        `num_episodes` episodes worth of experience using the current policy
        (which is initially random), then fit or re-fit the Q-function. Return
        the full set of episode data."""
        episodes = []
#        progress = tqdm(range(num_refits), file=sys.stdout,desc='num_refits')
        for i in range(num_refits):
#            progress.update(1)
#            episodes = []
            for _ in range(num_episodes):
                episodes.append(self.run_episode())
                ###
                real_episodes=self.run_episode(eps=0)
                R=real_episodes[2]
                S=real_episodes[0]
                A_S=real_episodes[3]
                best_r=0
                best_r2=0
                for j in range(len(R)):
                    best_r+=R[j]
                for j in range(len(A_S)):
                    best_r2+=A_S[j][1]
                print('Round: {}-{} Reward 1: {} Reward 2:{} B"{}'.format(i,_,best_r,best_r2,S[-1][3]))   
                ###
#            print(real_episodes[1])
            self.fit_Q(episodes=episodes, discount=discount)
            if save:
#                with open('./fqi-regressor-iter-{}.pkl'.format(i+1), 'wb') as f:
                with open('./fqi-regressor-iter.pkl', 'wb') as f:
                    pickle.dump(self.regressor, f)
     
                    
#        progress.close()
        # S=episodes[0]
        # R=episodes[2]
        # best_r=0
        # best_r2=0
        # for i in range(len(R)):
        #     best_r2+=R[i]*discount**i
        #     best_r+=S[i][1]
        # print(best_r)
        # print(best_r2)
        return episodes

    def encode_o(self, states, actions):
        """Encode states and actions as a single input array, suitable for
        passing into the `fit` method of a scikit-learn model."""
        if isinstance(actions, int):
            actions = [actions] * len(states)
        actions = np.array([[a] for a in actions])
        states = np.array(states)
        return np.hstack([states, actions])

    def encode(self, states, actions):
        """Encode states and actions as a single input array, suitable for
        passing into the `fit` method of a scikit-learn model."""
        if isinstance(actions[0], int):
            actions = [actions] * len(states)
        states = np.array(states)
        return np.hstack([states, actions])



if __name__ == '__main__':
    print('Here goes nothing')
    discount=1
    num_refits=5
    num_episodes=100
#    episode_length=10
    First_time=True
    lam=0.0
    if First_time:
        env=FittedQIteration()
        episodes=env.fit( num_refits=num_refits, num_episodes=num_episodes,discount=discount)
     
        with open('Result_{}_SIR_refits={}_episodes={}.pickle'.format(lam,num_refits,num_episodes), 'wb') as f:
                    pickle.dump([env,episodes], f)
    else:            
        with open('Result_{}_SIR_refits={}_episodes={}.pickle'.format(lam,num_refits,num_episodes), 'rb') as f:
            X = pickle.load(f)  
        env=X[0]
        episodes=X[1]
    
    
    real_episodes=env.run_episode(eps=0)
    S=real_episodes[0]
    A=real_episodes[1]
    R=real_episodes[2]
    A_S=real_episodes[3]
    best_r=0
    for i in range(len(R)):
         best_r+=R[i]*discount**i
        # best_r+=S[i][0]+S[i][2]
#        best_r+=R[i]

    random_action_episodes=env.run_episode(eps=1)
    S_r=random_action_episodes[0]
    A_r=random_action_episodes[1]
    R_r=random_action_episodes[2]
    A_S_r=random_action_episodes[3]
    random_r=0
    for i in range(len(R_r)):
         random_r+=R_r[i]*discount**i
        # random_r+=S_r[i][0]+S_r[i][2]
#        random_r+=R_r[i]
    
    state=real_episodes[0][0]
    
    estimate_r=env.Q([state])[0]
#    print("Estimate Reward(Discounted):")
#    print(np.max(estimate_r))
    print("Real Reward(Discounted):")
    print(best_r)
    print("Random Action Reward(Discounted):")
    print(random_r)
    I=[]
    action=[]
    for i in range(1,len(A_S)):
        action.append(4*(A_S[i-1][3]-A_S[i][3]))
    
    for i in range(len(A_S)):
        I.append(A_S[i][1])
    plt.plot(range(len(I)),I)
    I_r=[]
    for i in range(len(A_S_r)):
        I_r.append(A_S_r[i][1])
    plt.plot(range(len(I_r)),I_r)
    