import gym
from gym import spaces
import numpy as np

class CustomSIREnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	def __init__(self, R0=3, T_treat=30):
		super(CustomSIREnv, self).__init__()
		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using discrete actions:
		self.action_space = spaces.Discrete(4)
		self.observation_space = spaces.Dict({"S": spaces.Box(low=0, high=1, shape=(1,), dtype=np.double),
								 "I": spaces.Box(low=0, high=1, shape=(1,), dtype=np.double),
								 "R": spaces.Box(low=0, high=1, shape=(1,), dtype=np.double),
								 "B": spaces.Box(low=0, high=80, shape=(1,), dtype=np.double),
								 "t": spaces.Discrete(400),
								 "I_sum": spaces.Box(low=0, high=400, shape=(1,), dtype=np.double),
								 "switch_budgets": spaces.Discrete(5),
								 "p_action": spaces.Discrete(4),
								 "terminal": spaces.Discrete(2),
								 })

		self.T=400
		
		self.R0 = R0
		self.T_treat = T_treat
		self.T_base = self.T_treat/self.R0
		'''Intervention Relateted'''
		# self.minduration=10
		self.num_states=9

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
		return T_trans_c*self.T_base

	def is_action_illegal(self,ACTION):
		if(self.get_action_cost(ACTION)>=self.B):
			return True
		else:
			return False

	def _take_action(self, ACTION):
		# Set the current price to a random price within the time step
		# Init state variables
		T_trans=self.get_action_T(ACTION)
	   
		# Init derivative vector
		dydt = np.zeros(self.num_states)
		# Write differential equations
		dydt[0] = -self.I*self.S/(T_trans)
		dydt[1] = self.I*self.S/(T_trans) - self.I/self.T_treat
		dydt[2] = self.I/self.T_treat
		dydt[3] = -self.get_action_cost(ACTION)
		dydt[4] = 1
		dydt[5] = self.I+dydt[1]
		dydt[6] = -int(self.p_action == ACTION)
		dydt[7] = ACTION-self.p_action
		dydt[8] = int(self.t+dydt[4]>=self.T)

		self.S += dydt[0]
		self.I += dydt[1]
		self.R += dydt[2]
		self.B += dydt[3]
		self.t += dydt[4]
		self.I_sum += dydt[5]
		self.switch_budgets += dydt[6]
		self.p_action += dydt[7]
		self.terminal += dydt[8]


	def _next_observation(self):
		obs = []
		obs.append(self.S)
		obs.append(self.I)
		obs.append(self.R)
		obs.append(self.B)
		obs.append(self.t)
		obs.append(self.I_sum)
		obs.append(self.switch_budgets)
		obs.append(self.p_action)
		obs.append(self.terminal)
		obs = np.array(obs)
		return obs


	def calc_QALY_reward(self):
		reward=-self.I
		return reward
	
	def calc_burden_reward(self):
		reward=0
		if self.I>0.1:
			reward=-100*(max(self.I-0.1,0))
		return reward
	
	def calc_delay_reward(self):
		reward=self.t*self.I
		return reward


	def step(self, action):
	# Execute one time step within the environment

	# Execute one time step within the environment
		self._take_action(action)
		self.current_step += 1
		# reward = self.calc_QALY_reward()
		# reward = self.calc_delay_reward()
		reward = self.calc_burden_reward()

		done = self.t >= self.T
		obs = self._next_observation()
		return obs, reward, done, {}


	def reset(self):
	# Reset the state of the environment to an initial state
		# Reset the state of the environment to an initial state
		self.S = 0.999
		self.I = 0.001
		self.R = 0.0
		self.B = 80.0
		self.t = 0
		self.I_sum = 0.001
		self.switch_budgets = 5
		self.p_action = 0
		self.terminal = 0

		# Set the current step to a random point within the data frame
		self.current_step = 0

		return self._next_observation()


	def render(self, close=False):
	  # Render the environment to the screen
		print(f'Step: {self.current_step}')
		print(f'S: {self.S}')
		print(f'I: {self.I}')
		print(f'R: {self.R}')
		print(f'B: {self.B}')
		print(f't: {self.t}')
		print(f'I_sum: {self.I_sum}')
		print(f'switch_budgets: {self.switch_budgets}')
		print(f'p_action: {self.p_action}')
		print(f'terminal: {self.terminal}')
		
