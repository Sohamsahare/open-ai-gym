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
import time

env = gym.make('MountainCar-v0')
env.seed(1); torch.manual_seed(1); np.random.seed(1)

# hyperparameters
episode_count = 1000
buffer_size = 100000
max_steps = 200
epsilon = 0.3
tau = 0.003
learning_rate = 0.003
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
    def __init__(self, env, gamma, epsilon, tau, batch_size, learning_rate, hidden1 = 64, hidden2 = 32):
        # initialzing module class
        super(Model, self).__init__()
        
        # setting gym environment variables
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        # setting hyper parameters
        self.t_step = 0
        self.epsilon = epsilon
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        
        # builing local and target model for DDQN
        self.model, self.modelAdv, self.modelVal = self.build_model()
        self.target_model, self.target_modelAdv, self.target_modelVal = self.build_model()
        
        # setting loss and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        
        # updating target model to match local model
        self.soft_update()
        
        # setting up experience replay memory 
        self.memory = ReplayBuffer(self.action_size, buffer_size, batch_size, 1)
        
    def build_model(self):
        # traditional neural network with state space -> hidden layer 2 nodes
        l1 = nn.Linear(self.state_size, self.hidden1, bias = True)
        l2 = nn.Linear(self.hidden1, self.hidden2, bias = True)
        
        # advantage value computed using a nn 
        # with hidden layer 2 nodes -> action space
        adv1 = nn.Linear(self.hidden2, 16)
        adv2 = nn.Linear(16,self.action_size)
        
        # value computed using a nn
        # with hidden layer 2 -> 1
        val1 = nn.Linear(self.hidden2,16)
        val2 = nn.Linear(16,1)
        
        # Dueling DQN using advantage and value
        model = torch.nn.Sequential(
            l1,
            nn.ReLU(),
            l2,
        )
        
        modelAdv = torch.nn.Sequential(
            model,
            nn.ReLU(),
            adv1,
            adv2,
        )
        
        modelVal = torch.nn.Sequential(
            model,
            nn.ReLU(),
            val1,
            val2
        )
        
        return model, modelAdv, modelVal
    
    def predict_model(self, x, is_target = False):
        if not is_target:
            adv = self.modelAdv(x)
            val = self.modelVal(x)
        else:
            adv = self.target_modelAdv(x)
            val = self.target_modelVal(x)
        return adv + val - np.mean(adv.detach().numpy())        
    
    def soft_update(self):
        tau = self.tau
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
    def act(self, state):
        # Epsilon Greedy approach to choose action
        if np.random.rand(1) < self.epsilon :
            action = self.env.action_space.sample()
        else:
            state = Variable(torch.from_numpy(state).type(torch.FloatTensor))
            Q = self.predict_model(state)
            _, action = torch.max(Q,-1)
            action = action.item() 
        return action
    
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
        
        Q = self.predict_model(states)
        Q_target = Variable(Q.clone())
        a = self.predict_model(next_states)
        t = self.predict_model(next_states, is_target = True )
        
        for i in range(self.batch_size):
            if dones[i]:
                Q_target[i][actions[i]] = rewards[i]
            else:
                Q_target[i][actions[i]] = rewards[i] + torch.mul(self.gamma, t[i][np.argmax(a[i].detach().numpy())])
            
        # train model network
        loss = self.loss_fn(Q,Q_target)
        
        # # Updating weights according to the loss
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon-Greedy
        # reduce epsilon with every learning cycle
        if self.epsilon > self.epsilon_min:
           self.epsilon *= self.epsilon_decay
    
'''
# to encourage machine to go higher
max_position = -.4
positions = np.ndarray([0,2])
rewards = []
successful = []
# testing for 1000 episodes with random actions
print('Testing with random actions ->')
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
'''
# defining the model
model = Model( env, gamma, epsilon, tau, batch_size, learning_rate )

max_position = -0.4
has_solved = False
first_solve_at = -1

for episode in trange(episode_count):
    state = env.reset()
    score = 0
    
    for step in range(max_steps):
        # if episode % 100 == 0 and episode > 0:
            # env.render()
            
        # act according to policy following epsilon greedy approach
        action = model.act(state)    
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
                model.soft_update()
                if not has_solved:
                    first_solve_at = episode + 1
                    has_solved = True
                success_count += 1
                # Adjust learning rate
                model.scheduler.step()
            # loss_history.append(loss)
            reward_history.append(score)
            all_positions.append(next_state[0])
            break
        else:
            state = next_state
            
# model.load_state_dict(torch.load('dqn_mountain_car_parameters.model'))
# model = torch.load('dqn_mountain_car_model.model')
print('First solved at -> ',first_solve_at)
print('Succesful episodes -> ',success_count)
print('Success Rate => {.2f} %'.format(success_count/episode_count * 100))

for episode in range(5):
    s = env.reset()
    score = 0
    for step in range(200):
        env.render()
        action = model.act(s)
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

# plot policy to show which actions are taken
X = np.random.uniform(-1.2, 0.6, 10000)
Y = np.random.uniform(-0.07, 0.07, 10000)
Z = []

# getting output values for each possible state
for i in range(len(X)):
    _, temp = torch.max(model.model(Variable(torch.from_numpy(np.array([X[i],Y[i]]))).type(torch.FloatTensor)), dim =-1)
    z = temp.item()
    Z.append(z)
Z = pd.Series(Z)
colors = {0:'blue',1:'lime',2:'red'}
colors = Z.apply(lambda x:colors[x])
labels = ['Left','Right','Nothing']

# plot the policy as patches
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

torch.save(model.state_dict,'ddqn_mountain_car_parameters.model')
torch.save(model,'ddqn_mountain_car_model.model')