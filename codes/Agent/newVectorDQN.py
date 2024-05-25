import sys
sys.path.append('/content/drive/My Drive/Minesweeper [RL]/codes')

import torch
import torch.nn as nn
import torch.optim as optim
import copy 

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
LEARN_EPOCH = 50000
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

class Agent:
    def __init__(self, env, net, **kwargs):
        self.env = env

        # Environment Settings
        self.mem_size = kwargs.get("MEM_SIZE")
        self.mem_size_min = kwargs.get("MEM_SIZE_MIN")

        # Learning Settings
        self.batch_size = kwargs.get("BATCH_SIZE")
        self.learning_rate = kwargs.get("LEARNING_RATE")
        self.learn_decay = kwargs.get("LEARN_DECAY")
        self.learn_epoch = kwargs.get("LEARN_EPOCH")
        # self.learn_min = kwargs.get("LEARN_MIN")
        self.discount = kwargs.get("DISCOUNT")

        # Exploration Settings
        self.epsilon = kwargs.get("EPSILON")
        self.epsilon_decay = kwargs.get("EPSILON_DECAY")
        self.epsilon_min = kwargs.get("EPSILON_MIN")

        # loss 
        self.loss_fn = nn.MSELoss()
        self.losses = []

        # target net update
        self.target_update_counter = 0
        self.update_target_baseline = kwargs.get("UPDATE_TARGET_EVERY")

        # def model 
        self.model = copy.deepcopy(net)
        self.target_model = copy.deepcopy(net)

        self.target_model.load_state_dict(self.model.state_dict())

        self.model.to(device)
        self.target_model.to(device)

        # optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.learn_epoch, gamma = self.learn_decay)

        # replay memory
        self.replay_memory = deque(maxlen=self.mem_size)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_action(self, state):
        '''
        get_action은 하나의 state_img만을 받는다.
        '''
        if np.random.random() < self.epsilon:
            # take random action
            action = np.random.choice(range(self.env.total_tiles))

        else:
            self.model.eval()

            with torch.no_grad():
                state = torch.tensor(state.reshape(1,1,self.env.nrows,self.env.ncols),
                                     dtype=torch.float32).to(device)
                total_action = self.model(state).view(-1)
                total_action = total_action.cpu()

                self.total_action = total_action
                
                action = torch.argmax(total_action).item()

        return action

    def train(self, done):
        if len(self.replay_memory) < self.mem_size_min:
            return

        # 리플레이 메모리에서 배치 사이즈만큼 데이터를 꺼낸다.
        # batch[i] = (current_state, action, reward, new_current_state, done)
        batch = random.sample(self.replay_memory, self.batch_size)

        # 배치 안에 저장되어 있는 정보 꺼내기
        current_states, actions, rewards, next_states, epi_dones = zip(*batch)

        current_states =  torch.tensor(np.array(current_states), dtype=torch.float32).reshape(-1,1,self.env.nrows,self.env.ncols).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).reshape(-1,1,self.env.nrows,self.env.ncols).to(device)

        actions = torch.tensor(np.array(actions), dtype=torch.int).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).reshape(-1,1).to(device)
        epi_dones = torch.tensor(np.array(epi_dones), dtype=torch.float).reshape(-1,1).to(device)

        self.model.train()
        self.target_model.eval()

        current_q_values = self.model(current_states)

        with torch.no_grad():
            next_q_values = self.target_model(next_states)

        target_value = rewards + (1 - epi_dones) * self.discount * torch.max(next_q_values, dim=1)[0].reshape(-1,1)
        target_value = target_value.flatten()

        target_q_values = copy.deepcopy(current_q_values.detach())
        target_q_values[range(BATCH_SIZE), actions] = target_value

        cost = self.loss_fn(current_q_values, target_q_values)

        running_loss = cost.item()

        self.losses.append(round(running_loss,6))

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        if done:
            self.target_update_counter += 1
            self.scheduler.step()

        if self.target_update_counter == self.update_target_baseline:
            self.update_target_model()
            self.target_update_counter = 0

        # decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)


class Limited18Agent(Agent):
    def __init__(self, env, net, replay_memory=False, **kwargs):
        super().__init__(env, net, **kwargs)
        # 불러올 리플레이 메모리가 있다면 불러옴
        if replay_memory:
            self.replay_memory = replay_memory

    def update_replay_memory(self, transition):
        current_state = transition[0]

        if np.sum(current_state != self.env.unrevealed) >= 18: # 경험적인 데이터 18(나름 하이퍼파라미터긴 함ㅋ)
            self.replay_memory.append(transition)


# agent = Agent(env, 
#               MEM_SIZE=MEM_SIZE,
#               MEM_SIZE_MIN=MEM_SIZE_MIN,
#               BATCH_SIZE=BATCH_SIZE,
#               LEARNING_RATE=LEARNING_RATE,
#               LEARN_DECAY=LEARN_DECAY,
#               LEARN_MIN=LEARN_MIN,
#               DISCOUNT=DISCOUNT,
#               EPSILON=EPSILON,
#               EPSILON_DECAY=EPSILON_DECAY,
#               EPSILON_MIN=EPSILON_MIN,
#               UPDATE_TARGET_EVERY=UPDATE_TARGET_EVERY)
            
# agent = Limited18Agent(env, 
#                       MEM_SIZE=MEM_SIZE,
#                       MEM_SIZE_MIN=MEM_SIZE_MIN,
#                       BATCH_SIZE=BATCH_SIZE,
#                       LEARNING_RATE=LEARNING_RATE,
#                       LEARN_DECAY=LEARN_DECAY,
#                       LEARN_MIN=LEARN_MIN,
#                       DISCOUNT=DISCOUNT,
#                       EPSILON=EPSILON,
#                       EPSILON_DECAY=EPSILON_DECAY,
#                       EPSILON_MIN=EPSILON_MIN,
#                       UPDATE_TARGET_EVERY=UPDATE_TARGET_EVERY)