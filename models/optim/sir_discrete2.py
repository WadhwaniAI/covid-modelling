import numpy as np
import random
import matplotlib.pyplot as plt

class SIR_Discrete(object):
    def __init__(self, S=0.99, I=0.01, R=0, B=30,days_last=0,switch_budgets=6,pAction=0):
        self.STATE=np.array([S, I, R, B,days_last,switch_budgets,pAction])
        self.R0 = 2.2 
        self.T_inf = 2.9
        self.T_trans = self.T_inf/self.R0
        self.T_treat = 15
        # self.N = 7e6
        # self.I0 = 1.0
        self.t=0
        '''Intervention Relateted'''
        self.num_actions=5
        self.num_states=7
    def reset(self):
        self.STATE=np.array([0.99, 0.01, 0, 30,0,6,0])
        self.t=0
    
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
            cost=0.75
        elif ACTION==4:
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
            T_trans_c=1.75
        elif ACTION==4:
            T_trans_c=2 
        return T_trans_c*self.T_inf/self.R0    
    
    def perform(self,ACTION):
        STATE_=self.STATE.copy()
        if self.is_action_legal(ACTION)==False:
            ACTION=0
        if ACTION!=0:
            if self.STATE[4]<10: #10 days duration
                ACTION=self.STATE[6]
                STATE_[5]=self.STATE[5]
                STATE_[4]=self.STATE[4]+1
            else: #After 10 days
                if ACTION!=self.STATE[6]: #different action
                    if STATE_[5]>0: #has switching budget
                        STATE_[5]=self.STATE[5]-1
                        STATE_[4]=0
                    else: #out of switching budget
                        if self.is_action_legal(ACTION)==True: #Remains same if possible
                            STATE_[5]=self.STATE[5]
                            STATE_[4]=self.STATE[4]+1
                            ACTION=self.STATE[6]
                        else:
                            STATE_[5]=self.STATE[5]
                            STATE_[4]=0
                            ACTION=0
                else: #Same action
                    STATE_[4]=self.STATE[4]+1
                    STATE_[5]=self.STATE[5]
        else: #no intervention
            STATE_[4]=0
            STATE_[5]=self.STATE[5]
        # 0:S, 1:E, 2:I, 3:B 4.days_last 5.switch_budgets 6.pAction
        T_trans=self.get_action_T(ACTION)
        STATE_[0] = self.STATE[0]-self.STATE[1]*self.STATE[0]/(T_trans)
        STATE_[1] = self.STATE[1]+self.STATE[1]*self.STATE[0]/(T_trans) - self.STATE[1]/self.T_treat
        STATE_[2] = self.STATE[2]+self.STATE[1]/self.T_treat
        STATE_[3] = self.STATE[3]-self.get_action_cost(ACTION)
        STATE_[6]=ACTION
        self.STATE=STATE_.copy()
        self.t+=1
        r=self.calc_reward()
        return r, self.STATE

    def calc_reward(self):
        S_coeficeint=1
        I_coeficeint=0
        R_coeficeint=1
        B_coeficeint=0
        coeficeint=np.array([S_coeficeint,I_coeficeint,R_coeficeint,B_coeficeint,0,0,0])
        reward=sum(coeficeint[:]*self.STATE[:])
        return reward
    
    
if __name__ == '__main__':
    N = 1e5
    I0 = 3.0
    env=SIR_Discrete((N - I0)/N, I0/N, 0, 30)
    # 0:S, 1:I, 2:R, 3:B
    S=[]
    R=[]
    I=[]
    for i in range(400):
        env.perform(0)
        state=env.STATE
        S.append(state[0])
        I.append(state[1])
        R.append(state[2])
    plt.plot(range(len(S)),S)
    plt.plot(range(len(I)),I)
    plt.plot(range(len(R)),R)
