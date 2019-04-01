from game import Board
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

num_episodes = 60000
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 1000

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)  
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Agent(object):
    """docstring for Agent"""
    def __init__(self):
        super(Agent, self).__init__()
        self.policy_net = DQN(9, 9).to(device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(75000)
        self.steps_done = 0
        self.num_episodes = 60000
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.95
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        
    def select_action(self, state, valid_actions):
        valid_actions = torch.tensor([1*valid_actions], device=device, dtype=torch.float32)
        global steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action = self.policy_net(state)*valid_actions
                return action.max(1)[1].view(1, 1)
        else:
            random_act = torch.tensor(np.random.rand(9), device=device, dtype=torch.float32)
            random_act = random_act*valid_actions
            return random_act.max(1)[1].view(1, 1)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        # print([batch.next_state])
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        print("Starting actual training")
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.policy_net(non_final_next_states).max(1)[0].detach()
        #print(next_state_values)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

env = Board()
done = True

player1 = Agent()
player2 = Agent()

for i in range(num_episodes):
    state = env.reset()
    for step in range(5):
        valid_actions = env.show_valid()
        state = state.ravel()
        state = torch.tensor([state], device=device, dtype=torch.float32)
        action = player1.select_action(state, valid_actions)
        next_state, reward, done, info = env.step1(action)
        if reward is 1:
            print("Player1 wins match {}".format(i))
        reward = torch.tensor([reward], device=device)
        next_state = next_state.ravel()
        if done:
            next_state = None
        else:
            next_state = next_state.ravel()
            next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
        player1.memory.push(state, action, next_state, reward)
        state = next_state

        if done:
            if reward is 0:
                print("Match {} is a draw".format(i))
            break

        valid_actions = env.show_valid()
        state = state.view(-1)
        action = player2.select_action(state, valid_actions)
        next_state, reward, done, info = env.step2(action)
        if reward is 1:
            print("Player2 wins match {}".format(i))
        reward = torch.tensor([reward], device=device)
        next_state = next_state.ravel()
        if done:
            next_state = None
        player2.memory.push(state, action, next_state, reward)
        state = next_state

        if done:
            if reward is 0:
                print("Match {} is a draw".format(i))
            break
    print("Optimizing")
    player1.optimize_model()
    player2.optimize_model()