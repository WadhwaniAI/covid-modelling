import numpy as np
import random
import matplotlib.pyplot as plt
import math

class SEIR_Discrete(object):
    def __init__(self, S=0.999, E=0, I=0.001, D_E=0, D_I=0, R_mild=0,R_severe_home=0,R_severe_hosp=0,R_fatal=0,C=0,D=0, B=80,t=0,switch_budgets=5,terminal=False):
        
        self.R0=3
        self.T_inf=3.3
        self.T_inc=4.5
        self.T_hosp=5
        self.T_recov_fatal=32
        self.P_severe=0.8
        self.P_fatal=0.11
        self.T_recov_severe=26
        self.T_recov_mild=11
        self.N=7e6
        self.init_infected=1
        self.q=0
        self.theta_E=0
        self.psi_E=1
        self.theta_I=0
        self.psi_I=1
        self.initialisation='starting'
        self.observed_values=None
        self.E_hosp_ratio=0.5 #####
        self.I_hosp_ratio=0.39 #####
        self.factor=(1.77/1.23) #####

        self.T=400
        self.H=0
        """
        This class implements SEIR + Hospitalisation + Severity Levels + Testing 
        The model further implements 
        - pre, post, and during lockdown behaviour 
        - different initialisations : intermediate and starting 

        The state variables are : 

        S : No of susceptible people
        E : No of exposed people
        I : No of infected people
        D_E : No of exposed people (detected)
        D_I : No of infected people (detected)
        R_mild : No of people recovering from a mild version of the infection
        R_severe_home : No of people recovering from a severe version of the infection (at home)
        R_severe_hosp : No of people recovering from a fatal version of the infection (at hospital)
        R_fatal : No of people recovering from a fatal version of the infection
        C : No of recovered people
        D : No of deceased people 

        The sum total is is always N (total population)

        """

        """
        The parameters are : 

        R0 values - 
        pre_lockdown_R0: R0 value pre-lockdown (float)
        lockdown_R0: R0 value during lockdown (float)
        post_lockdown_R0: R0 value post-lockdown (float)

        Transmission parameters - 
        T_inc: The incubation time of the infection (float)
        T_inf: The duration for which an individual is infectious (float)

        Probability of contracting different types of infections - 
        P_mild: Probability of contracting a mild infection (float - [0, 1])
        P_severe: Probability of contracting a severe infection (float - [0, 1])
        P_fatal: Probability of contracting a fatal infection (float - [0, 1])

        Clinical time parameters - 
        T_recov_mild: Time it takes for an individual with a mild infection to recover (float)
        T_recov_severe: Time it takes for an individual with a severe infection to recover (float)
        T_hosp: Time it takes for an individual to get hospitalised, after they have been diagnosed (float)
        T_recov_fatal: Time it takes for an individual with a fatal infection to die (float)

        Testing Parameters - 
        'q': Perfection of quarantining 
        If q = 0, quarantining is perfect. q = 1. quarantining is absolutely imperfect
        'theta_E': Percentage of people in the Exposed bucket that are tested daily
        'psi_E': Sensitivity of test that Exposed people undergo
        'theta_I': Percentage of people in the Infected bucket that are tested daily
        'psi_I': Sensitivity of test that Infected people undergo

        Lockdown parameters - 
        starting_date: Datetime value that corresponds to Day 0 of modelling (datetime/str)
        lockdown_day: Number of days from the starting_date, after which lockdown is initiated (int)
        lockdown_removal_day: Number of days from the starting_date, after which lockdown is removed (int)

        Misc - 
        N: Total population
        initialisation : method of initialisation ('intermediate'/'starting')
        E_hosp_ratio : Ratio for Exposed to hospitalised for initialisation
        I_hosp_ratio : Ratio for Infected to hospitalised for initialisation
        """
        self.P_mild = 1 - self.P_severe - self.P_fatal

        # define testing related parameters
        self.T_inf_detected = self.T_inf
        self.T_inc_detected = self.T_inc

        self.P_mild_detected = self.P_mild
        self.P_severe_detected = self.P_severe
        self.P_fatal_detected = self.P_fatal
        
        
        self.STATE=np.array([S, E, I, D_E, D_I, R_mild,R_severe_home,R_severe_hosp,R_fatal,C,D, B,t,switch_budgets,terminal])
        '''Intervention Relateted'''
        self.minduration=10
        self.duration_scale=5
        self.num_actions=4
        self.max_duration=40
        self.num_states=7
        
        
        
        
        
    def reset(self):
        self.H=0
        self.STATE=np.array([0.999, 0, 0.001, 0, 0, 0,0,0,0,0,0, 80,0,5,False])
        # self.STATE=np.array([0.999, 0.001, 0, 80,0,5,False])
    
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
        return T_trans_c    
    
    
    def perform_leader(self, Strength, Duration):
        a_s=[]
        reward=0
        Duration=Duration*self.duration_scale+self.minduration
        Duration=int(Duration)
        if self.STATE[14]==False:
#            if self.STATE[3]<=0:
#                Strength=0
#            elif self.STATE[3]<self.get_action_cost(Strength)*Duration:
#                Duration=int(math.floor(float(self.STATE[3])/self.get_action_cost(Strength)))+1
            if self.STATE[12]+Duration>self.T:
                Duration=int(self.T-self.STATE[4])
            for _ in range(Duration):
                r,self.STATE=self.perform(Strength)
                a_s.append(self.STATE)
                reward+=r
            if Strength>0:
                self.STATE[13]-=1
        if self.STATE[12]>=self.T:
            self.STATE[14]=True
        return reward, self.STATE, a_s
    
    def perform(self,ACTION):
        STATE_=self.STATE.copy()
        # [S, E, I, D_E, D_I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D]
        #################
        self.T_inf_D=self.T_inf 
        self.T_inc_D=self.T_inc 
        self.P_mild_D=self.P_mild
        self.P_severe_D=self.P_severe
        self.P_fatal_D=self.P_fatal
        #################
        
        
        T_trans=self.get_action_T(ACTION)*self.T_inf/self.R0
        T_trans_D=self.get_action_T(ACTION)*self.T_inf_D/self.R0

        

        STATE_=self.STATE.copy()     
        STATE_[0] += - STATE_[2] * self.STATE[0] / (T_trans) - (self.q / T_trans_D) * (self.STATE[0] * self.STATE[4]) # S
        STATE_[1] += STATE_[2] * self.STATE[0] / (T_trans) + (self.q / T_trans_D) * (self.STATE[0] * self.STATE[4]) - (self.STATE[1]/ self.T_inc) - (self.theta_E * self.psi_E * self.STATE[1]) # E
        STATE_[2] += self.STATE[1] / self.T_inc - STATE_[2] / self.T_inf - (self.theta_I * self.psi_I * STATE_[2]) # I
        STATE_[3] += (self.theta_E * self.psi_E * self.STATE[1]) - (1 / self.T_inc_D) * self.STATE[3] # D_E
        STATE_[4] += (self.theta_I * self.psi_I * STATE_[2]) + (1 / self.T_inc_D) * self.STATE[3] - (1 / self.T_inf_D) * self.STATE[4] # D_I 
        STATE_[5] += (1/self.T_inf)*(self.P_mild*STATE_[2]) + (1/self.T_inf_D)*(self.P_mild_D*self.STATE[4]) - self.STATE[5]/self.T_recov_mild # R_mild
        STATE_[6] += (1/self.T_inf)*(self.P_severe*STATE_[2]) + (1/self.T_inf_D)*(self.P_severe_D*self.STATE[4]) - self.STATE[6]/self.T_hosp # R_severe_home
        STATE_[7] += self.STATE[6]/self.T_hosp - self.STATE[7]/self.T_recov_severe # R_severe_hosp
        STATE_[8] += (1/self.T_inf)*(self.P_fatal*STATE_[2]) + (1/self.T_inf_D)*(self.P_fatal_D*self.STATE[4]) - self.STATE[8]/self.T_recov_fatal # R_fatal
        STATE_[9] += self.STATE[5]/self.T_recov_mild + self.STATE[7]/self.T_recov_severe # C
        STATE_[10] += self.STATE[8]/self.T_recov_fatal # D
        STATE_[11] += -self.get_action_cost(ACTION)
        STATE_[12] += 1
        self.STATE=STATE_.copy()
        

        # r=self.calc_QALY_reward()
        # r=self.calc_burden_reward()
        # r=self.calc_peak_reward()
        r=self.calc_delay_reward()
        return r, self.STATE

    def calc_QALY_reward(self):
        # reward=-self.STATE[1]*1000
        reward=1-self.STATE[2]
        return reward
    
    def calc_burden_reward(self):
        reward=0
        if self.STATE[2]>0.1:
#            reward=-1000
            reward=-1000*(self.STATE[2]-0.1)
        return reward
    def calc_peak_reward(self):
        reward=0
        if self.STATE[2]>self.H:
            reward=-1000*(self.STATE[2]-self.H)
            self.H=self.STATE[2]
        return reward
    
    def calc_delay_reward(self):
        reward=self.STATE[12]*self.STATE[2]
        return reward
    
    
if __name__ == '__main__':
    N = 1e5
    I0 = 100.0
    env=SEIR_Discrete(B=80)
    S=[]
    E=[]
    I=[]
    for i in range(400):
        env.perform(0)
        state=env.STATE
        S.append(state[0])
        E.append(state[1])
        I.append(state[2])
#        R.append(state[2])
    plt.plot(range(len(S)),S)
    plt.plot(range(len(E)),E)
    plt.plot(range(len(I)),I)

