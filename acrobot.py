import tensorflow as tf
import numpy as np
import random
import gym
import pickle
from statistics import median, mean
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn

env = gym.make('Acrobot-v1')
env.reset()

lr = 0.002
goal_steps = 200
initial_games = 10000

scores = []
accepted_scores = []
min_accepted_speed_1 = 1
min_accepted_speed_2 = 3

def some_random_games():
	Xmax = 0
	Ymax = 0
	countX = 0
	countY = 0
	for game in range(3):
		env.reset()
		for step in range(200):
			env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			x = np.abs(observation[4])
			y = np.abs(observation[5])
			if x > Xmax: 
				Xmax = x
				countX+=1
			if y > Ymax:
				Ymax = y
				countY+=1
			if done:
				break
	print(Xmax, countX, Ymax, countY)

some_random_games()

def mean_observation_at_index(observations, index):
	data = 0
	for obs in observations:
		data += np.abs(obs[0][index])
	return data / len(observations)

def generate_training_data():
	training_data = []
	revision_data = []
	l_max = 0
	h_max = 0
	speed_1 = 0
	speed_2 = 0
	
	for episode in range(initial_games):
		if episode % 1000 == 0:
			print('Episode ',episode+1,' has started')
		moves_made = []
		prev_obs = []
		max_speed_1 = 0
		max_speed_2 = 0
		add_to_revision = False
		for step in range(goal_steps):
			# env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if len(prev_obs) > 0:
				moves_made.append([prev_obs,action])
			max_speed_1 = np.abs( observation[4] )
			max_speed_2 = np.abs( observation[5] )
			if h_max < max_speed_1:
				h_max = max_speed_1
				add_to_revision = True
			if l_max < max_speed_2:
				l_max = max_speed_2
				add_to_revision = True
			prev_obs = observation
			if done:
				break
		speed_1 = mean_observation_at_index(moves_made,4)
		speed_2 = mean_observation_at_index(moves_made,5)
		if speed_1 > min_accepted_speed_1 or speed_2 > min_accepted_speed_2:
			for move in moves_made:
				if move[1] == 0:
					output = [0,1,0]
				elif move[1] == 1:
					output = [0,0,1]
				else:
				# move[1] == -1
					output = [1,0,0]
				if add_to_revision:
					revision_data.append([move[0], output])
				training_data.append([move[0], output])
		env.reset()
	
	print('Revision data -> ',len(revision_data))
	training_data = training_data + revision_data
	
	# just in case you wanted to reference later
	training_data_save = np.array(training_data)
	print(len(training_data_save))
	np.save('saved_acrobot_revision.npy',training_data_save)
	
	# some stats here, to further illustrate the neural network magic!
	print('Average Rotational Speed ',speed_1, speed_2)
	print('Max Rotational Speed were -> ',h_max,l_max)
	return training_data
	
def neural_network_model(input_size):
	network = input_data(shape=[None, input_size, 1], name='input')

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 3, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')
	model = tflearn.DNN(network, tensorboard_dir='log')
	return model
	
def train_model(training_data, model=False):
	print('training data ',training_data[0])
	
	X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
	y = [i[1] for i in training_data]
	
	if not model:
		model = neural_network_model(input_size = len(X[0]))
		
	model.fit({'input': X}, {'targets': y}, n_epoch=1, snapshot_step=100, show_metric=True, run_id='openai_learning')
	return model

def map_3_to_2(list):
	revamped_list = []
	map = {-1 : [1,0], 0:[0,0], 1:[0,1]}
	for y in list:
		revamped_list.append([y[0],map.get(np.argmax(y[1]))])
	return revamped_list
		
# training_data = generate_training_data()
print('Reading from file')
training_data = np.load('saved_acrobot_revision.npy')
# training_data = map_3_to_2(training_data)
print('File loaded. Size is  -> ',len(training_data))
print('Training data is \n',training_data)
model = train_model(training_data)
choices = []
sin_bottom = 0
sin_top = 0
for each_game in range(10):
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range(goal_steps):
		env.render()
		if len(prev_obs) == 0:
			action = random.randrange(0,2)
		else:
			action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0]) - 1
		choices.append(action)
		new_observation, reward, done, info = env.step(action)
		if np.abs(new_observation[0]) > sin_bottom:
			sin_bottom = new_observation[0]
		if np.abs(new_observation[2]) > sin_top:
			sin_top = new_observation[2]
		prev_obs = new_observation
		game_memory.append([new_observation, action])
		if done: break
print('choice 1:{}	choice 0:{} choice -1: {}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices),choices.count(-1)/len(choices)))
print('Highest angles achieved ',sin_bottom,sin_top)
# pickle.dump(model,open('rl_acro.pickle','wb'))
