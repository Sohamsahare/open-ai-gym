import gym
import numpy as np
import random
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.8196, # optimum = .8196, changing this seems have no influence
)

# setting up environment
env = gym.make('FrozenLake8x8-v0')

# possible actions and states of our environment
# Possible q-learning -> STATES		 | 	     ACTIONS
Q = np.zeros([env.observation_space.n, env.action_space.n])

# hyper parameters
# learning rate
lr = 0.8
# dicounting rate ( gamma )
y = 0.95
# number of episodes to train for
num_episodes = 15000
# list to store all rewards during training
rewards_list = []

# beginning q-value calculation using q-learning algorithm
for i in range(num_episodes):
	# reset the environment before training
	s = env.reset()
	# inital reward value for the episode
	total_rewards = 0
	# is the game over? ofc not, hence False
	d = False
	# current time step
	step = 0
	while step < 99:
		step +=1
		# for initial episodes, as all the q values will be zero,  
		# we add an offset to help agent explore the environment
		# as episodes increases, randomness offset decreases due 
		# to inverse proportion and q-values are taken into account more and more
		a = np.argmax(Q[s,:]+ np.random.rand(1,env.action_space.n)*(1./(i+1)))
		# exploration / exploitation trade off -------------------------^^^
		# take the action 
		s1,r,d,_ = env.step(a)
		# update q value as per bellman's eqn
		Q[s,a] += lr*(r+y*np.max(Q[s1,:])-Q[s,a])
		# append reward for the action
		total_rewards += r
		# new state is the old state for the next iteration
		s =s1
		# if reached H or G, end game
		if d == True:
			break
	rewards_list.append(total_rewards)

print('Score over time ',str(sum(rewards_list)/num_episodes))
print(Q)

# testing out our q table
env.reset()
# 5 different variations with slippery floor
for episode in range(5):
	state = env.reset()
	step = 0
	done = False
	print('*'*60)
	print('Episode ', episode)
	for step in range(99):
		# the highest value in a current state row 
		# corresponds to what action shoul be taken
		action = np.argmax(Q[state,:])
		new_state, reward, done, info = env.step(action)
		if done:
		# render the final move where the game had ended
			env.render()
			print('Number of steps ',step)
			break
		state = new_state
env.close()
