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

# class 
class DQNAgent:
    def __init__(self, env, conv_units=64):
        self.env = env

        # Deep Q-learning parameters
        self.discount = DISCOUNT
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON

        self.model = DQN(input_dims=self.env.state.shape,
                         n_actions=self.env.total_tiles,
                         conv_units=conv_units)

        self.target_model = DQN(input_dims=self.env.state.shape,
                                n_actions=self.env.total_tiles,
                                conv_units=conv_units)

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
        idx_arr = np.arange(self.env.total_tiles)
        unsolved = idx_arr[present_board == self.env.unrevealed]

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

                self.total_action = total_action
                
                action = torch.argmax(total_action).item()

        return action

    def train(self, done):
        if len(self.replay_memory) < MEM_SIZE_MIN:
            return

        # 리플레이 메모리에서 배치 사이즈만큼 데이터를 꺼낸다.
        # batch[i] = (current_state, action, reward, new_current_state, done)
        batch = random.sample(self.replay_memory, BATCH_SIZE)

        # 배치 안에 저장되어 있는 정보 꺼내기
        current_states, _, _, next_states, _ = zip(*batch)

        current_states =  torch.tensor(np.array(current_states), dtype=torch.float32).reshape(-1,1,self.env.nrows,self.env.ncols).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).reshape(-1,1,self.env.nrows,self.env.ncols).to(device)

        self.model.eval()
        self.target_model.eval()

        with torch.no_grad():
            current_q_values = self.model(current_states).reshape(-1,self.env.total_tiles).cpu().detach().tolist()
            next_q_values = self.target_model(next_states).cpu().detach().numpy()

        #  current_q_values를 target value가 되도록 업데이트하는 코드
        for index, (_, action, reward, _, epi_done) in enumerate(batch):
            if not epi_done:
                max_future_q = np.max(next_q_values[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_q_values[index][action] = new_q

        # train model
        self.model.train()

        x = current_states.to(device)
        y = torch.tensor(np.array(current_q_values), dtype=torch.float32).to(device)

        y_est = self.model(x)

        cost = self.loss_fn(y_est, y)

        running_loss = cost.item()
        self.losses.append(round(running_loss,6))

        self.optimizer.zero_grad()
        cost.backward()
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


class Limited18DQNAgent(DQNAgent):
    def __init__(self, env, conv_units=64, replay_memory=False):
        super().__init__(env, conv_units)
        # 불러올 리플레이 메모리가 있다면 불러옴
        if replay_memory:
            self.replay_memory = replay_memory

    def update_replay_memory(self, transition):
        current_state = transition[0]

        if np.sum(current_state != self.env.unrevealed) >= 18: # 경험적인 데이터 18(나름 하이퍼파라미터긴 함ㅋ)
            self.replay_memory.append(transition)