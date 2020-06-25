import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
sys.path.append('../..')
from models.optim.actor_critic_env import CustomSIREnv
from utils.optim.sir.objective_functions import *
from copy import deepcopy

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=1, metavar='G',
					help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
					help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
					help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
					help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = CustomSIREnv()
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
	"""
	implements both actor and critic in one model
	"""
	def __init__(self):
		super(Policy, self).__init__()
		self.affine1 = nn.Linear(9, 128)

		# actor's layer
		self.action_head = nn.Linear(128, 4)

		# critic's layer
		self.value_head = nn.Linear(128, 1)

		# action & reward buffer
		self.saved_actions = []
		self.rewards = []

	def forward(self, x):
		"""
		forward of both actor and critic
		"""
		x = F.relu(self.affine1(x))

		# actor: choses action to take from state s_t 
		# by returning probability of each action
		action_prob = F.softmax(self.action_head(x), dim=-1)

		# critic: evaluates being in the state s_t
		state_values = self.value_head(x)

		# return values for both actor and critic as a tuple of 2 values:
		# 1. a list with the probability of each action over the action space
		# 2. the value from state s_t 
		return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
	
	state = torch.from_numpy(state).float()
	probs, state_value = model(state)
	# print(probs)

	# create a categorical distribution over the list of probabilities of actions
	m = Categorical(probs)

	# and sample an action using the distribution
	action = m.sample()

	if env.is_action_illegal(action):
		action=torch.tensor(0)

	# save to action buffer
	model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

	# the action to take (left or right)
	return action.item()


def finish_episode():
	"""
	Training code. Calculates actor and critic loss and performs backprop.
	"""
	R = 0
	saved_actions = model.saved_actions
	policy_losses = [] # list to save actor (policy) loss
	value_losses = [] # list to save critic (value) loss
	returns = [] # list to save the true values

	# calculate the true value using rewards returned from the environment
	for r in model.rewards[::-1]:
		# calculate the discounted value
		R = r + args.gamma * R
		returns.insert(0, R)

	returns = torch.tensor(returns)
	returns = (returns - returns.mean()) / (returns.std() + eps)

	for (log_prob, value), R in zip(saved_actions, returns):
		advantage = R - value.item()

		# calculate actor (policy) loss 
		policy_losses.append((-log_prob * advantage).type(torch.float32))
		# print((-log_prob * advantage).type(torch.float32))
		# print(value, torch.tensor([R], dtype=torch.float32))
		# calculate critic (value) loss using L1 smooth loss
		value_losses.append(F.mse_loss(value, torch.tensor([R], dtype=torch.float32)))

	# reset gradients
	optimizer.zero_grad()

	# sum up all the values of policy_losses and value_losses
	loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
	print(loss)

	# perform backprop
	loss.backward()
	optimizer.step()

	# reset rewards and action buffer
	del model.rewards[:]
	del model.saved_actions[:]


def cost(array):
	add = 0
	for i in range(len(array)):
		add += env.get_action_cost(array[i])
	return add


def main():

	# run inifinitely many episodes
	for i_episode in count(1):

		# reset environment and episode reward
		state = env.reset()
		ep_reward = 0
		infection_array = []
		action_array = []
		# for each episode, only run 9999 steps so that we don't 
		# infinite loop while learning
		for t in range(1, 400):

			# select action from policy
			action = select_action(state)

			# take the action
			state, reward, done, _ = env.step(action)
			infection_array.append(state[1])
			action_array.append(action)

			if args.render:
				env.render()

			model.rewards.append(reward)
			ep_reward += reward
			if done:
				break


		infection_array = np.array(infection_array)
		action_array = np.array(action_array)


		# perform backprop
		finish_episode()

		# log results
		if i_episode % args.log_interval == 0:
			print('Episode {}\tLast reward: {:.2f}'.format(
				  i_episode, ep_reward))
			print('AUC-I: {}'.format(inf_auc(infection_array)))
			print('Height: {}'.format(inf_height(infection_array)))
			print('Time: {}'.format(inf_time(infection_array)))
			print('Burden: {}'.format(inf_burden(infection_array)))
			# print(infection_array)
			print(action_array)
			print('Budget Remaining: {}'.format(state[3]))
		if(inf_auc(infection_array)<=24):
			break





if __name__ == '__main__':
	main()