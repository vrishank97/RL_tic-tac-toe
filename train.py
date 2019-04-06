from board import Board
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

num_episodes = 100000
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 8000

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
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, num_classes)  
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent(object):
    """docstring for Agent"""
    def __init__(self):
        super(Agent, self).__init__()
        self.policy_net = DQN(9, 9).to(device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(2000)
        self.steps_done = 0
        self.num_episodes = 100000
        self.BATCH_SIZE = 256
        self.GAMMA = 0.98
        self.EPS_START = 0.95
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        
    def select_action(self, state, valid_actions):
        state = torch.tensor([state], device=device, dtype=torch.float32).view(-1)
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
        # # print("starting actual training")
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # # print("Continuing actual training")
        #print(non_final_next_states)
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
        action = player1.select_action(state, valid_actions)
        next_state1, reward1, done, info = env.step1(action)
        if done:
            next_state = None
        temp = next_state
        if not done:
            temp = torch.tensor([next_state.ravel()], device=device, dtype=torch.float32)
        player1.memory.push(torch.tensor([state.ravel()], device=device, dtype=torch.float32), action, temp, torch.tensor([reward], device=device))
        state = next_state

        if done:
            break

        valid_actions = env.show_valid()
        action = player2.select_action(state, valid_actions)
        next_state2, reward2, done, info = env.step2(action)
        if done:
            next_state = None
        temp = next_state
        if not done:
            temp = torch.tensor([next_state.ravel()], device=device, dtype=torch.float32)
        player2.memory.push(torch.tensor([state.ravel()], device=device, dtype=torch.float32), action, temp, torch.tensor([reward], device=device))
        state = next_state

        if done:
            break
    player1.optimize_model()
    player2.optimize_model()

print("Saving model 1")
torch.save(player1.policy_net, "player1.pth")
torch.save(player1.policy_net.state_dict(), "player1dict.pth")
print("Saving model 2")
torch.save(player2.policy_net, "player2.pth")
torch.save(player2.policy_net.state_dict(), "player2dict.pth")
