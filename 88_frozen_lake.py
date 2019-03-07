import gym
import numpy as np
import random
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.8196, # optimum = .8196, changing this seems have no influence
)

# setting up environment
env = gym.make('FrozenLakeNotSlippery-v0')

# possible actions and states of our environment
# for q learning 
action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size,action_size))
# print(qtable)

# defining hypre parameters
total_episodes = 15000
learning_rate = 0.8
# max steps per episode
max_steps = 99
# discounting rate
gamma = 0.95

# exploration rate
epsilon = 1.0
# exploration probablity at start
max_epsilon = 1.0
# min exploration probablity
min_epsilon = 0.01
# exponential decay rate for exploration probablity
decay_rate = 0.005

# list of rewards
rewards = []
for episode in range(total_episodes):
	# reset the env
	state = env.reset()
	step = 0
	done = False
	total_rewards = 0
	for step in range(max_steps):
		# choose an action a in the current world state (s)
		# first we randomize a number
		exp_exp_tradeoff = random.uniform(0,1)
		# if this number > epsilon -> exploit else explore 
		if exp_exp_tradeoff > epsilon:
			action = np.argmax(qtable[state,:])
		else:
			action = env.action_space.sample()
		new_State, reward, done, info = env.step(action)
		# Update !(s,a) := Q(s,a) + lr*[R(s,a) + gamma * max(Q(s',a')) - Q(s,a)]
		# qtable[new_State,:] : all the actions we can take from new state
		qtable[state, action] +=  learning_rate * ( reward + gamma * np.max(qtable[new_State,:]) - qtable[state,action])
		total_rewards += reward
		state = new_State
		# if dead, end episode
		if done == True:
			break
	# reduce epsilon ( because we need less and less exploration)
	epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode )
	rewards.append(total_rewards)
print('Score over time: '+ str(sum(rewards)/total_episodes))
print(qtable)

####################################################################################################################################################################
# to run 

env.reset()
for episode in range(5):
	state = env.reset()
	step = 0
	done = False
	print('*'*60)
	print('Episode ', episode+1)
	for step in range(max_steps):
		action = np.argmax(qtable[state,:])
		new_state, reward, done, info = env.step(action)
		if done:
			env.render()
			print('Number of steps ',step)
			break
		state = new_state
env.close()

