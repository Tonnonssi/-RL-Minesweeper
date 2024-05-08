import sys
sys.path.append('/content/drive/My Drive/Minesweeper [RL]/codes')

from Net.basicNet import *

import torch
import torch.nn as nn
import torch.optim as optim

import random 
import numpy as np
from collections import deque

# Environment settings
MEM_SIZE = 50000 
MEM_SIZE_MIN = 1000 

# Learning settings
BATCH_SIZE = 64
LEARNING_RATE = 0.01
LEARN_DECAY = 0.99975 
LEARN_MIN = 0.001
DISCOUNT = 0.1 

# Exploration settings
EPSILON = 0.95
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.01

# DQN settings
CONV_UNITS = 64 
UPDATE_TARGET_EVERY = 5

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDQNAgent:
    def __init__(self, env, conv_units=64, dense_units=256):
        self.env = env

        # Deep Q-learning parameters
        self.discount = DISCOUNT
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON

        self.model = DQN(input_dims=self.env.state.shape,
                         n_actions=self.env.total_tiles,
                         conv_units=conv_units,
                         dense_units=dense_units)

        self.target_model = DQN(input_dims=self.env.state.shape,
                                n_actions=self.env.total_tiles,
                                conv_units=conv_units,
                                dense_units=dense_units)

        self.target_model.load_state_dict(self.model.state_dict())

        self.replay_memory = deque(maxlen=MEM_SIZE)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-4)

        self.model.to(device)
        self.target_model.to(device)

        self.target_update_counter = 0

        self.losses = []

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_action(self, state):
        '''
        get_action은 하나의 state_img만을 받는다.
        '''
        present_board = state.reshape(self.env.total_tiles) # flatten to get move idx
        unsolved = [i for i,x in enumerate(present_board) if x == self.env.unrevealed]

        if np.random.random() < self.epsilon:
            # take random action
            action = np.random.choice(unsolved)

        else:
            self.model.eval()

            with torch.no_grad():
                state = torch.tensor(state.reshape(1,1,self.env.nrows,self.env.ncols),
                                         dtype=torch.float32).to(device)
                total_action = self.model(state).view(-1)
                total_action = total_action.cpu()

                # 이미 오픈한 타일은 move 대상에서 제외된다.
                total_action[present_board != self.env.unrevealed] = torch.min(total_action)

                action = torch.argmax(total_action).item()

        return action

    def train(self, done):
        if len(self.replay_memory) < MEM_SIZE_MIN:
            return

        self.model.train()
        self.target_model.eval()

        # 리플레이 메모리에서 배치 사이즈만큼 데이터를 꺼낸다.
        # batch[i] = (current_state, action, reward, next_state, done)
        batch = random.sample(self.replay_memory, BATCH_SIZE)

        # 배치 안에 저장되어 있는 정보 꺼내기
        current_states, batched_actions, batched_rewards, next_states, batched_dones = zip(*batch)

        # state 정의
        current_states = torch.tensor(np.array(current_states), dtype=torch.float32).reshape(-1,1,self.env.nrows,self.env.ncols).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).reshape(-1,1,self.env.nrows,self.env.ncols).to(device)

        action_batch = torch.tensor(batched_actions).reshape(1,-1).to(device) # reshape 안해주면 index로써 사용할 수 없다.
        reward_batch = torch.tensor(batched_rewards).to(device)
        done_batch = torch.tensor(batched_dones, dtype=torch.float32).to(device) # bool -> 0/1

        # 예측값
        pred_q_values = self.model(current_states).gather(1, action_batch) # action idx의 데이터만 꺼냄

        # 일시적 정답 역할 - DDQN
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1).reshape(1,-1) # reshape 안해주면 index로써 사용할 수 없다.
            next_q_values = self.target_model(next_states).gather(1, next_actions)

            target_q_values = reward_batch + (torch.ones(next_q_values.shape, device=device) - done_batch) * self.discount * next_q_values

        loss = self.loss_fn(pred_q_values, target_q_values)

        running_loss = loss.item()
        self.losses.append(round(running_loss,6))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if done:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

        # decay learning rate
        self.learning_rate = max(LEARN_MIN, self.learning_rate*LEARN_DECAY)

        # decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)