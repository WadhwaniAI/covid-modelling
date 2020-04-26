import numpy as np
import random
import matplotlib.pyplot as plt

class SEIR_Discrete(object):
    def __init__(self, S, E, I, R_mild, R_severe, R_severe_hosp, R_fatal, C, D,B):
        self.STATE=[S, E, I, R_mild, R_severe, R_severe_hosp, R_fatal, C, D,B]
        self.R0 = 2.2 
        self.T_inf = 2.9
        self.T_trans = self.T_inf/self.R0
        self.T_inc = 5.2
        self.T_recov_mild = (14 - self.T_inf)
        self.T_hosp = 5
        self.T_recov_severe = (31.5 - self.T_inf)
        self.T_death = 32    
        self.P_severe = 0.2
        self.P_fatal = 0.02
        self.P_mild = 1 - self.P_severe - self.P_fatal
        # self.N = 7e6
        # self.I0 = 1.0
        self.t=0
        
    def reset(self):
        self.STATE=[S, E, I, R_mild, R_severe, R_severe_hosp, R_fatal, C, D,B]
        self.t=0
    
    def is_action_legal(self,ACTION):
        if(self.get_action_cost(ACTION)<=self.STATE[9]):
            return True
        else:
            return False
        
    def get_action_cost(self,ACTION):
        
        return 0
    
    def get_action_T(self,ACTION):
    
        return self.T_inf/self.R0    
    
    def perform(self,ACTION):
        if self.is_action_legal(ACTION):
            STATE_=self.STATE.copy()
        else:
            print("Exceed Budget")
        # 0:S, 1:E, 2:I, 3:R_mild, 4:R_severe, 5:R_severe_hosp, 6:R_fatal, 7:C, 8:D, 9:B
        T_trans=self.get_action_T(ACTION)
        STATE_[0] = self.STATE[0]-self.STATE[2]*self.STATE[0]/(T_trans)
        STATE_[1] = self.STATE[1]+self.STATE[2]*self.STATE[0]/(T_trans) - self.STATE[1]/self.T_inc
        STATE_[2] = self.STATE[2]+self.STATE[1]/self.T_inc - self.STATE[2]/self.T_inf
        STATE_[3] = self.STATE[3]+(self.P_mild*self.STATE[2])/self.T_inf - self.STATE[3]/self.T_recov_mild
        STATE_[4] = self.STATE[4]+(self.P_severe*self.STATE[2])/self.T_inf - self.STATE[4]/self.T_hosp
        STATE_[5] = self.STATE[5]+self.STATE[4]/self.T_hosp - self.STATE[5]/self.T_recov_severe
        STATE_[6] = self.STATE[6]+(self.P_fatal*self.STATE[2])/self.T_inf - self.STATE[6]/self.T_death
        STATE_[7] = self.STATE[7]+self.STATE[3]/self.T_recov_mild + self.STATE[5]/self.T_recov_severe
        STATE_[8] = self.STATE[8]+self.STATE[6]/self.T_death
        STATE_[9] = self.STATE[9]-self.get_action_cost(ACTION)
        self.STATE=STATE_.copy()
        self.t+=1

    def calc_reward(self):
        S_coeficeint=1
        E_coeficeint=0
        I_coeficeint=0
        R_mild_coeficeint=0
        R_severe_coeficeint=0
        R_severe_home_coeficeint=0
        R_R_fatal_coeficeint=0
        C_coeficeint=1
        D_coeficeint=0
        coeficeint=[S_coeficeint,E_coeficeint,I_coeficeint,R_mild_coeficeint,R_severe_coeficeint,R_severe_home_coeficeint,R_severe_home_coeficeint,R_R_fatal_coeficeint,C_coeficeint,D_coeficeint]
        reward=sum(coeficeint[:]*self.STATE[:])
        return reward
    
    
if __name__ == '__main__':
    N = 1000
    I0 = 1.0
    env=SEIR_Discrete((N - I0)/N, 0, I0/N, 0, 0, 0, 0, 0, 0,100,)
    # 0:S, 1:E, 2:I, 3:R_mild, 4:R_severe, 5:R_severe_hosp, 6:R_fatal, 7:C, 8:D, 9:B
    S=[]
    E=[]
    I=[]
    for i in range(400):
        env.perform(0)
        state=env.STATE
        S.append(state[0])
        E.append(state[1])
        I.append(state[2])
    plt.plot(range(len(S)),S)
    plt.plot(range(len(E)),E)
    plt.plot(range(len(I)),I)
