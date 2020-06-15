import numpy as np
import random
import matplotlib.pyplot as plt

class SEIR_Discrete(object):
    def __init__(self, S=0.99, E=0, I=0.01, R_mild=0, R_severe=0, R_severe_hosp=0, R_fatal=0, C=0, D=0,B=30,days_last=0,switch_budgets=5,pAction=0):
        self.STATE=np.array([S, E, I, R_mild, R_severe, R_severe_hosp, R_fatal, C, D,B,days_last,switch_budgets,pAction])
        self.R0 = 2.2 
        self.T_inf = 2.9
        self.T_trans = self.T_inf/self.R0
        self.T_inc = 5.2
        self.T_recov_mild = (14 - self.T_inf)
        self.T_hosp = 5
        self.T_recov_severe = (31.5 - self.T_inf)
        self.T_recov_fatal = 32    
        self.P_severe = 0.2
        self.P_fatal = 0.02
        self.P_mild = 1 - self.P_severe - self.P_fatal
        # self.N = 7e6
        # self.I0 = 1.0
        self.t=0
        '''Intervention Relateted'''
        self.num_actions=4
        self.num_states=13
    def reset(self):
        self.STATE=np.array([0.99, 0, 0.01, 0, 0, 0, 0, 0, 0,30,0,5,0])
        self.t=0
    
    def is_action_legal(self,ACTION):
        if(self.get_action_cost(ACTION)<=self.STATE[9]):
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
            T_trans_c=1.25
        elif ACTION==2:
            T_trans_c=1.5 
        elif ACTION==3:
            T_trans_c=2 
        return T_trans_c*self.T_inf/self.R0    
    
    def perform(self,ACTION):
        STATE_=self.STATE.copy()
        if self.is_action_legal(ACTION)==False:
            ACTION=0
            # 0:S, 1:E, 2:I, 3:R_mild, 4:R_severe, 5:R_severe_hosp, 6:R_fatal, 7:C, 8:D, 9:B,
            # 10:days_last, 11:switch_budgets,12:pAction
        if (ACTION!=0 or self.STATE[12]!=0) and self.is_action_legal(ACTION)==True:
            if self.STATE[12]==0:
                STATE_[10]=1
                STATE_[11]=self.STATE[11]-1
            else:
                if self.STATE[10]<11 or ACTION==self.STATE[12]: #10 days duration or same action
                    STATE_[10]=self.STATE[10]+1
                    STATE_[11]=self.STATE[11]
                    ACTION=self.STATE[12]
                else: #After 10 days and different action
                    if self.STATE[11]>0: #has switching budget
                        STATE_[10]=1
                        STATE_[11]=self.STATE[11]-1
                    else: #out of switching budget
                        STATE_[10]=0
                        STATE_[11]=self.STATE[11]
                        ACTION=0
        else: #no intervention
            STATE_[10]=0
            STATE_[11]=self.STATE[11]
            ACTION=0
        STATE_[12]=ACTION
        T_trans=self.get_action_T(ACTION)
        STATE_[0] = self.STATE[0]-self.STATE[2]*self.STATE[0]/(T_trans)
        STATE_[1] = self.STATE[1]+self.STATE[2]*self.STATE[0]/(T_trans) - self.STATE[1]/self.T_inc
        STATE_[2] = self.STATE[2]+self.STATE[1]/self.T_inc - self.STATE[2]/self.T_inf
        STATE_[3] = self.STATE[3]+(self.P_mild*self.STATE[2])/self.T_inf - self.STATE[3]/self.T_recov_mild
        STATE_[4] = self.STATE[4]+(self.P_severe*self.STATE[2])/self.T_inf - self.STATE[4]/self.T_hosp
        STATE_[5] = self.STATE[5]+self.STATE[4]/self.T_hosp - self.STATE[5]/self.T_recov_severe
        STATE_[6] = self.STATE[6]+(self.P_fatal*self.STATE[2])/self.T_inf - self.STATE[6]/self.T_recov_fatal
        STATE_[7] = self.STATE[7]+self.STATE[3]/self.T_recov_mild + self.STATE[5]/self.T_recov_severe
        STATE_[8] = self.STATE[8]+self.STATE[6]/self.T_recov_fatal
        STATE_[9] = self.STATE[9]-self.get_action_cost(ACTION)
        self.STATE=STATE_.copy()
        self.t+=1
        r=self.calc_reward()
        return r, self.STATE

    def calc_reward(self):
        S_coeficeint=1
        E_coeficeint=0.5
        I_coeficeint=0
        R_mild_coeficeint=1
        R_severe_coeficeint=1
        R_severe_home_coeficeint=1
        R_R_fatal_coeficeint=1
        C_coeficeint=1
        D_coeficeint=0
        coeficeint=np.array([S_coeficeint,E_coeficeint,I_coeficeint,R_mild_coeficeint,R_severe_coeficeint,R_severe_home_coeficeint,R_R_fatal_coeficeint,C_coeficeint,D_coeficeint,0,0,0,0])
        reward=sum(coeficeint[:]*self.STATE[:])
        return reward
    
    
if __name__ == '__main__':
    N = 1000
    I0 = 1.0
    env=SEIR_Discrete((N - I0)/N, 0, I0/N, 0, 0, 0, 0, 0, 0,50,0,0,0)
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
