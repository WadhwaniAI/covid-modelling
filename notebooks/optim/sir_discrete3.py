import numpy as np
import random
import matplotlib.pyplot as plt
import math

class SIR_Discrete(object):
    def __init__(self, S=0.999, I=0.001, R=0, B=60,t=0,switch_budgets=5,terminal=False):
        self.T=400
        
        self.STATE=np.array([S, I, R, B,t,switch_budgets,terminal])
        self.R0 = 3
        self.T_treat = 30
        self.T_trans = self.T_treat/self.R0
        # self.T_treat = 15
        # self.N = 1e5
        # self.I0 = 100.0
        '''Intervention Relateted'''
        self.minduration=10
        self.duration_scale=10
        self.num_actions=4
        self.max_duration=20
        self.num_states=7
    def reset(self):
        self.STATE=np.array([0.999, 0.001, 0, 60,0,5,False])
    
    def is_action_legal(self,ACTION):
        if(self.get_action_cost(ACTION)<=self.STATE[3]):
            return True
        else:
            return False
        
    def get_action_cost(self,ACTION):
        if ACTION==0:
            cost=0
        elif ACTION==1:
            cost=0.25
        elif ACTION==2:
            cost=0.5    
        elif ACTION==3:
            cost=1       
        return cost
    
    def get_action_T(self,ACTION):
        if ACTION==0:
            T_trans_c=1
        elif ACTION==1:
            T_trans_c=1.5
        elif ACTION==2:
            T_trans_c=2 
        elif ACTION==3:
            T_trans_c=3 
        return T_trans_c*self.T_treat/self.R0    
    
    
    def perform_leader(self, Strength, Duration):
        a_s=[]
        reward=0
#        Duration=+self.minduration
        Duration=Duration*self.duration_scale+self.minduration
        Duration=int(Duration)
#        print(self.STATE[4])
#        print(Duration)
        if self.STATE[6]==False:
#            if self.STATE[3]<=0:
#                Strength=0
#            elif self.STATE[3]<self.get_action_cost(Strength)*Duration:
#                Duration=int(math.floor(float(self.STATE[3])/self.get_action_cost(Strength)))+1
            if self.STATE[4]+Duration>self.T:
                Duration=int(self.T-self.STATE[4])
            for _ in range(Duration):
                r,self.STATE=self.perform(Strength)
                a_s.append(self.STATE)
                reward+=r
            if Strength>0:
                self.STATE[5]-=1
        if self.STATE[4]>=self.T:
            self.STATE[6]=True
#        print(self.STATE[4])
        return reward, self.STATE, a_s
    
    def perform(self,ACTION):
        STATE_=self.STATE.copy()
        # 0:S, 1:I, 2:R, 3:B 4:t 5:switch_budgets 6:pAction
        T_trans=self.get_action_T(ACTION)
        STATE_[0] = self.STATE[0]-self.STATE[1]*self.STATE[0]/(T_trans)
        STATE_[1] = self.STATE[1]+self.STATE[1]*self.STATE[0]/(T_trans) - self.STATE[1]/self.T_treat
        STATE_[2] = self.STATE[2]+self.STATE[1]/self.T_treat
        STATE_[3] = self.STATE[3]-self.get_action_cost(ACTION)
        STATE_[4]+=1
        self.STATE=STATE_.copy()
        r=self.calc_burden_reward()
        return r, self.STATE

    def calc_QALY_reward(self):
        reward=(1-self.STATE[1])/400
        return reward
    
    def calc_burden_reward(self):
        reward=0
        if self.STATE[1]>0.15:
            reward=-100
#            reward=-100*(self.STATE[1]-0.15)
        return reward
    
    def calc_delay_reward(self):
        reward=self.STATE[4]*self.STATE[1]
        return reward
    
    
if __name__ == '__main__':
    N = 1e5
    I0 = 100.0
    env=SIR_Discrete((N - I0)/N, I0/N, 0, 30)
    # 0:S, 1:I, 2:R, 3:B
    S=[]
    R=[]
    I=[]
    for i in range(400):
        env.perform(0)
        state=env.STATE
#        S.append(state[0])
        I.append(state[1])
#        R.append(state[2])
#    plt.plot(range(len(S)),S)
    plt.plot(range(len(I)),I)
#    plt.plot(range(len(R)),R)
    reward=0
    for i in range(len(I)):
        reward+=I[i]
    print(reward)
