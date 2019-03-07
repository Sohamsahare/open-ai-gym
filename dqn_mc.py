import matplotlib.pyplot as plt
import gym
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from replay_buffer import ReplayBuffer

env = gym.make('MountainCar-v0')
env.seed(1); torch.manual_seed(1); np.random.seed(1)

        
episode_count = 1000
buffer_size = 100000
max_steps = 200
epsilon = 0.3
tau = 0.001
learning_rate = 0.001
gamma = 0.99
batch_size = 32
state = env.reset()
success_count = 0
loss_history = []
reward_history = []
all_positions = []
max_position = -0.4
update_every = 4

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.hidden1 = 64
        self.hidden2 = 64
        self.l1 = nn.Linear(self.state_size, self.hidden1, bias = True)
        self.l2 = nn.Linear(self.hidden1, self.hidden2, bias = True)
        self.l3 = nn.Linear(self.hidden2, self.action_size, bias = True)
        self.memory = ReplayBuffer(self.action_size, buffer_size, batch_size, 1)
        self.t_step = 0
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.tau = tau
        self.batch_size = batch_size
        
    def forward(self,x):
        model = torch.nn.Sequential(
            self.l1,
            self.l2,
            self.l3
        )
        return model(x)
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % update_every
        if self.t_step == 0:
            if len(self.memory) > batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def learn( self, experiences):
        # sample of batch_size from replay memory
        states, actions, rewards, next_states, dones = experiences
        
        # Q-value of current state and the next predicted state in the tuple
        Q = self(states)
        Q1 = self(next_states)
        
        # Max of such Q values 
        maxQ1, _ = torch.max(Q1,-1)
        
        # to update target Q value and calculate MSE Loss
        Q_target = Q.clone()
        Q_target = Variable(Q_target)
        for i in range(self.batch_size):
            if dones[i]:
                Q_target[i][actions[i]] = rewards[i]
            else:
            # Q(s,a) += r(s,a) + GAMMA * Max(Q(s',a'))
                Q_target[i][actions[i]] = rewards[i] + torch.mul(maxQ1[i].detach(), self.gamma)
        loss = self.loss_fn(Q,Q_target)
        
        # Updating weights according to the loss
        self.zero_grad()
        loss.backward()
        self.optimizer.step()

 
    
# to encourage machine to go higher
max_position = -.4
positions = np.ndarray([0,2])
rewards = []
successful = []
print('Testing with random actions ->')
# testing for 1000 episodes with random actions
for episode in trange(1000):
    running_reward = 0
    env.reset()
    done = False
    for i in range(200):
        state, reward, done, _ = env.step(np.random.randint(0,3))
        # Give a reward for reaching a new maximum position
        if state[0] > max_position:
            max_position = state[0]
            positions = np.append(positions, [[episode, max_position]], axis=0)
            running_reward += 10
        else:
            running_reward += reward
        if done: 
            if state[0] >= 0.5:
                successful.append(episode)
            rewards.append(running_reward)
            break

print('Furthest Position: {}'.format(max_position))
print('successful episodes: {}'.format(np.count_nonzero(successful)))

max_position = -0.4
# defining the model
model = Model()
loss_func = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(),lr = learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
# '''
has_solved = False
first_solve_at = -1
for episode in trange(episode_count):
    state = env.reset()
    score = 0
    loss = 0
    for step in range(max_steps):
        # if episode % 100 == 0 and episode > 0:
            # env.render()
        state_tensor = Variable(torch.from_numpy(state).type(torch.FloatTensor))
        Q = model(state_tensor)
        
        # Epsilon Greedy approach to choose action
        if np.random.rand(1) < epsilon :
            action = env.action_space.sample()
        else:
            _, action = torch.max(Q,-1)
            action = action.item()
        
        next_state, reward, done, info = env.step(action)
        
        # change reward structure for better performance in training
        # making it depend on current position and a multiple of it's current speed
        # increases
        reward = next_state[0] + np.abs(next_state[1] * 80)
        
        # updating the max_position for the current episode
        if next_state[0] > max_position:
            max_position = next_state[0]
            
        # increase reward for reaching exceptional heights    
        if next_state[0] >= 0.5:
            reward += 1

        # adding to replay memory and learning 
        model.step(state, action, reward, next_state, done)
        
        score += reward
        
        # to check if we passed the episode or not
        if done:
            if next_state[0] >= 0.5:
                if not has_solved:
                    first_solve_at = episode + 1
                    has_solved = True
                success_count += 1
                # Adjust epsilon
                epsilon *= .95
                # Adjust learning rate
                scheduler.step()
            loss_history.append(loss)
            reward_history.append(score)
            all_positions.append(next_state[0])
            break
            
        else:
            state = next_state
            
# '''
# model.load_state_dict(torch.load('dqn_mountain_car_parameters.model'))
# model = torch.load('dqn_mountain_car_model.model')
print('First solved at -> ',first_solve_at)
print('Succesful episodes -> ',success_count)
for episode in range(5):
    s = env.reset()
    score = 0
    for step in range(200):
        env.render()
        state = Variable(torch.from_numpy(s).type(torch.FloatTensor))
        Q = model(state)
        _, action = torch.max(Q,-1)
        action = action.item()
        s1,r,d,_ = env.step(action)
        r = s1[0] + np.abs(s1[1] * 80)
        if s1[0] >= 0.5:
            r += 1
        score += r
        
        if d:
            print('Score -> ',score)
            break
        else:
            s = s1
        
# plot episode vs max_position
plt.figure(1, figsize=[10,5])
plt.subplot(211)
plt.plot(all_positions)
plt.xlabel('Episode')
plt.ylabel('Furthest Position')
# plot episode vs reward at that episode
plt.subplot(212)
plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

X = np.random.uniform(-1.2, 0.6, 10000)
Y = np.random.uniform(-0.07, 0.07, 10000)
Z = []
for i in range(len(X)):
    _, temp = torch.max(model(Variable(torch.from_numpy(np.array([X[i],Y[i]]))).type(torch.FloatTensor)), dim =-1)
    z = temp.item()
    Z.append(z)
Z = pd.Series(Z)
colors = {0:'blue',1:'lime',2:'red'}
colors = Z.apply(lambda x:colors[x])
labels = ['Left','Right','Nothing']

import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
fig = plt.figure(5, figsize=[7,7])
ax = fig.gca()
plt.set_cmap('brg')
surf = ax.scatter(X,Y, c=Z)
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_title('Policy')
recs = []
for i in range(0,3):
     recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
plt.legend(recs,labels,loc=4,ncol=3)
fig.savefig('Policy - Modified.png')
plt.show()

# torch.save(model.state_dict,'dqn_mountain_car_parameters.model')
# torch.save(model,'dqn_mountain_car_model.model')

