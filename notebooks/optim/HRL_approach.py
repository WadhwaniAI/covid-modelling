import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
# sys.path.append('../..')
# from models.optim.SEIR_Discrete import SEIR_Discrete
#from models.optim.sir_discrete import SIR_Discrete
# from utils.optim.sir import objective_functions
from sir_discrete3 import SIR_Discrete
from seir_dis_RL import SEIR_Discrete
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
        self.simulator = SEIR_Discrete()
        # self.simulator = SIR_Discrete()
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
#                            Q[i,j,k]=400-state[4]
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
            #assign the Q value of illegal actions to zero 
            for i in range(Table.shape[0]):
                for j in range(Table.shape[1]):
                    Duration=j*self.simulator.duration_scale+self.simulator.minduration
                    if state[3]<self.simulator.get_action_cost(i)*Duration or state[4]+Duration>self.simulator.T:
                        Table[i][j]=0
            
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
        Duration=a[1]*self.simulator.duration_scale+self.simulator.minduration
        if state[3]<self.simulator.get_action_cost(a[0])*Duration:
            a[0]=0
        if state[4]+Duration>self.simulator.T:
            a[1]=math.floor((self.simulator.T-state[4])/10)
            
            
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
#            print(s[4])
            S.append(s)
            A.append(a)
            R.append(r)
            A_S=A_S+a_s
        EP_length=R.copy()
        for i in range(len(R)):
            EP_length[i]=len(R)-i-1
#        print(A)
        return S, A, R, A_S, EP_length

    def run_episode_na(self, eps=0.1):
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
#            a=self.policy(s, eps=eps)
#            print(a)
#            print('-----------')
            r, s_ , a_s= self.simulator.perform_leader(0,100)
            s=s_
#            print(s[4])
            S.append(s)
            A.append(0)
            R.append(r)
            A_S=A_S+a_s
        EP_length=R.copy()
        for i in range(len(R)):
            EP_length[i]=len(R)-i-1
#        print(A)
        return S, A, R, A_S, EP_length

    def fit_Q(self, episodes, num_iters=10, discount=0.9999):
        """Fit and re-fit the Q function using historical data for the
        specified number of `iters` at the specified `discount` factor"""
        S1 = np.vstack([ep[0][:-1] for ep in episodes])
        S2 = np.vstack([ep[0][1:] for ep in episodes])     
        A = np.vstack([ep[1] for ep in episodes])
        R = np.hstack([ep[2] for ep in episodes])
        L = np.hstack([ep[4] for ep in episodes])
        inputs = self.encode(S1, A)
#        progress = tqdm(range(num_iters), file=sys.stdout,desc='num_iters')
        for iters in range(num_iters):
#            progress.update(1)
#            targets = R + discount * ((self.Q(S2).max(axis=2)).max(axis=1))
            targets=R.copy()
            for i in range(len(R)):               
                targets[i]=sum(R[i:(i+L[i])])
#            alpha=1
#            targets = (self.Q(S1).max(axis=2)).max(axis=1)+alpha*(R + discount * (self.Q(S2).max(axis=2)).max(axis=1)-(self.Q(S1).max(axis=2)).max(axis=1))

            self.regressor.fit(inputs, targets)
#            self.regressor.fit(A, targets)
#        progress.close()
        
        
    def fit(self, num_refits=1, num_episodes=15, discount=0.9999, save=False):
        """Perform fitted-Q iteration. For `outer_iters` steps, gain
        `num_episodes` episodes worth of experience using the current policy
        (which is initially random), then fit or re-fit the Q-function. Return
        the full set of episode data."""
        Pretrain=False
        if(Pretrain):
#            PreA=[[0,22],[1,10],[0,41],[1,50],[2,130],[0,254]]
#            PreA=[[0,1],[1,0],[0,3],[1,4],[2,12],[0,24]]
            PreA=[[0,2],[1,0],[0,6],[1,8],[2,24],[0,49]]
            S=[]
            R=[]
            A_S=[]
            self.simulator.reset()
            s=self.simulator.STATE
            S.append(s)
            t=0
            while self.simulator.STATE[6]==False:
                a=PreA[t]
                t=t+1
                r, s_ , a_s= self.simulator.perform_leader(a[0],a[1])
                s=s_
                S.append(s)
                R.append(r)
                A_S=A_S+a_s
            PreEP=[]
            PreEP.append([S, PreA, R, A_S])
            S1 = np.vstack([ep[0][:-1] for ep in PreEP])
            A = np.vstack([ep[1] for ep in PreEP])
            inputs = self.encode(S1, A)
            targets=R.copy()
            for i in range(len(R)):
                targets[i]=sum(R[i:len(R)])
#            print(targets)
            self.regressor.fit(inputs, targets)
        episodes = []
        progress = tqdm(range(num_refits), file=sys.stdout,desc='num_refits')
        epsilon = 0.9
        decay = 0.5
        for i in range(num_refits):
            progress.update(1)
            episodes = []
            epsilon*=decay
            for _ in range(num_episodes):
                #Tune epsilon latter TODO
                epsilon*=decay
                # episodes.append(self.run_episode(eps=max(epsilon,0.01)))
                # episodes.append(self.run_episode(eps=1-(i/num_refits)))
                episodes.append(self.run_episode(eps=0.1))
                if len(episodes)%20==0:
                    self.fit_Q(episodes=episodes, discount=discount)
                    # episodes = []
                # self.fit_Q(episodes=episodes, discount=discount)

#                print('Round: {}-{}'.format(i,_))
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
            Q=0
            if is_fitted(self.regressor):
                Q=((self.Q(S).max(axis=2)).max(axis=1))[0]
            print('Reward 1: {} Reward 2:{} Estimate R:{}'.format(best_r,best_r2,Q))   
#            self.fit_Q(episodes=episodes, discount=discount)
            if save:
                with open('./fqi-regressor-iter-{}.pkl'.format(i+1), 'wb') as f:
                # with open('./fqi-regressor-iter.pkl', 'wb') as f:
                    pickle.dump(self.regressor, f)
     
                    
        progress.close()
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

def get_score(I, score_type):
    if score_type==0:
        Score=sum(I)
    elif score_type==1:
        Score = np.max(I)   
    elif score_type==2:
        auci = np.sum(I)
        running_sum = np.ones(len(I))
        time_array = np.ones(10)
        for i in range(len(I)):
            running_sum[i] = np.sum(I[:i+1])
        for i in range(1,11):
            time_array[i-1] = np.argmax(running_sum>=auci*0.1*i)+1
        Score = np.mean(time_array)
    elif score_type==3:
        Score = np.sum(I[I>=0.1])
    return(Score)



if __name__ == '__main__':
    print('Here goes nothing')
    discount=1
    num_refits=10
    num_episodes=1000
    lam=1
    First_time=True
    if First_time:
        env=FittedQIteration()
        episodes=env.fit( num_refits=num_refits, num_episodes=num_episodes,discount=discount)
     
        with open('Result_{}_SEIR_refits={}_episodes={}_delay.pickle'.format(lam,num_refits,num_episodes), 'wb') as f:
                    pickle.dump([env,episodes], f)
    else:            
        with open('Result_{}_SEIR_refits={}_episodes={}_delay.pickle'.format(lam,num_refits,num_episodes), 'rb') as f:
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
#         best_r+=R[i]*discount**i
        # best_r+=S[i][0]+S[i][2]
        best_r+=R[i]
#    RR=0
#    for i in range(len(A_S)):
#        RR+=A_S[i][1]*A_S[i][4]
#    print(RR)
    random_action_episodes=env.run_episode_na(eps=1)
    S_r=random_action_episodes[0]
    A_r=random_action_episodes[1]
    R_r=random_action_episodes[2]
    A_S_r=random_action_episodes[3]
    random_r=0
    for i in range(len(R_r)):
#         random_r+=R_r[i]*discount**i
        # random_r+=S_r[i][0]+S_r[i][2]
        random_r+=R_r[i]
    
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

    print("QALY")
    print("NI_Score="+str(get_score(np.array(I_r), 0)))
    print("Score="+str(get_score(np.array(I), 0)))
    print("Peak")
    print("NI_Score="+str(get_score(np.array(I_r), 1)))
    print("Score="+str(get_score(np.array(I), 1)))
    print("Delay")
    print("NI_Score="+str(get_score(np.array(I_r), 2)))
    print("Score="+str(get_score(np.array(I), 2)))
    print("Burden")
    print("NI_Score="+str(get_score(np.array(I_r), 3)))
    print("Score="+str(get_score(np.array(I), 3)))