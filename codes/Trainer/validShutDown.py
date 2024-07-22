import sys
sys.path.append('/content/drive/My Drive/Minesweeper [RL]/codes')

import time 
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

EPISODES = 50000
PRINT_INTERVAL = 100
TRAIN_RENDER = False

TRAIN_TIMESTEPS = ['every timestep', 'every episodes']
TRAIN_TIMESTEP = TRAIN_TIMESTEPS[0]
VISUAL_INTERVAL = 100

VALID_SAMPLE = 1000
VALID_INTERVAL = 10

class Trainer:
    def __init__(self, env, agent, tester_agent, name, train_start=True, **kwargs):
        self.env = env
        self.agent = agent

        self.progress_list = []
        self.wins_list = []
        self.ep_rewards_list = []

        self.name = name
        self.f_path = '/content/drive/MyDrive/Minesweeper [RL]/models'
        self.total_path = self.f_path + '/' + self.name
        self.tester_agent = tester_agent

        self.best_model_train = None
        self.best_model_valid = None
        self.best_model_successed = None

        self.baseline_train = 0
        self.baseline_valid = 0
        self.baseline_successed = 0

        self.simple_valid = 0

        # Parameters
        self.episodes = kwargs.get("EPISODES")

        self.print_interval = kwargs.get("PRINT_INTERVAL")
        self.train_render = kwargs.get("TRAIN_RENDER")
        self.train_timestep = kwargs.get("TRAIN_TIMESTEP")

        self.valid_sample = kwargs.get("VALID_SAMPLE")
        self.valid_interval = kwargs.get("VALID_INTERVAL")

        self.visual_interval = kwargs.get("VISUAL_INTERVAL") if kwargs.get("VISUAL_INTERVAL") is not None else self.print_interval
        self.interval = 500

        if train_start:
            self.train()
            self.visualize_train()
            self.save_model()

    def train(self):

        start = time.time()

        win_rate = 0
        valid_win_rate = 0
        successed_win_rate = 0

        for episode in range(self.episodes):
            self.env.reset()

            n_clicks = 0
            done = False
            episode_reward = 0

            while not done:
                current_state = self.env.state

                action = self.agent.get_action(current_state)

                next_state, reward, done = self.env.step(action)

                episode_reward += reward

                self.agent.update_replay_memory((current_state, action, reward, next_state, done))

                if self.train_timestep == TRAIN_TIMESTEPS[0]: # every timestep
                    self.agent.train(done)

                n_clicks += 1

            if self.train_timestep == TRAIN_TIMESTEPS[1]: # every episodes
                self.agent.train(done)

            if self.train_render:
                self.env.render(self.env.state)
                print(episode_reward)

            self.progress_list.append(n_clicks)
            self.ep_rewards_list.append(episode_reward)
            self.wins_list.append(reward == self.env.rewards['win'])

            # 승리한 모델 저장 
            if reward == self.env.rewards['win']:
                successed_state = self.agent.model.state_dict()

            if (episode+1) % self.print_interval == 0:
                med_progress = np.median(self.progress_list[-self.print_interval:])
                win_rate = np.sum(self.wins_list[-self.print_interval:]) / self.print_interval
                med_reward = np.median(self.ep_rewards_list[-self.print_interval:])

                print(f"Episode: [{self.episodes}/{episode+1}]| Median progress: {med_progress:.2f} | Median reward: {med_reward:.2f} | Win rate : {win_rate:.3f} | Epsilon: {self.agent.epsilon:.2f}")

                if win_rate > self.baseline_train:
                    self.baseline_train = win_rate
                    self.best_model_train = self.agent.model.state_dict()

                    self.simple_valid = 10

                    # 지난 구간의 승률이 baseline을 뛰어넘었을 때만 구간 중 승리한 모델을 점검한다. 
                    print("valid latest successed model")
                    successed_win_rate = self.valid_model(self.env, self.tester_agent, episode, self.valid_sample, successed_state)

                    if successed_win_rate > self.baseline_successed:
                        self.best_model_successed = successed_state
                        self.baseline_successed = successed_win_rate

            if self.simple_valid > 0:
                valid_state = self.agent.model.state_dict()
                valid_win_rate = self.valid_model(self.env, self.tester_agent, episode, self.valid_sample, valid_state)
                self.simple_valid -= 1

                if valid_win_rate > self.baseline_valid:
                    self.baseline_valid = valid_win_rate
                    self.best_model_valid = self.agent.model.state_dict()

        print(round(time.time() - start, 2))

    def valid_model(self, env, agent, episode, epoch, model_state):

        progress_list, wins_list, ep_rewards = [], [], []

        agent.epsilon = 0.0 # valid에서는 탐험을 꺼준다.

        agent.model.load_state_dict(model_state)
        agent.target_model.load_state_dict(model_state)

        for i in range(epoch):

            env.reset()

            done = False
            n_clicks = 0
            episode_reward = 0

            while not done:
                current_state = env.state

                action = agent.get_action(current_state)

                next_state, reward, done = env.step(action)
                
                if (current_state == next_state).all(): # 같은 곳을 계속 누르는 상황을 탈출시키는 ShutDown Code
                    done = True

                episode_reward += reward
                n_clicks += 1

            progress_list.append(n_clicks)
            ep_rewards.append(episode_reward)
            wins_list.append(reward == env.rewards['win'])

        print(f"Valid n:{epoch}, Median progress: {np.median(progress_list):.2f}, Median reward: {np.median(ep_rewards):.2f}, Win rate : {np.sum(wins_list)/len(wins_list)}")

        return np.sum(wins_list)/len(wins_list) # 승률을 반환한다.

    def visualize_train(self, progress=True, win_rates=True, rewards=True, losses=True):
        progresses = []
        win_rates = []
        rewards = []
        losses = []

        for start in range(0, len(self.progress_list)-self.visual_interval, self.visual_interval):
            progresses.append(sum(self.progress_list[start:start+self.visual_interval]) / self.visual_interval)
            win_rates.append(sum(self.wins_list[start:start+self.visual_interval]) / self.visual_interval)
            rewards.append(sum(self.ep_rewards_list[start:start+self.visual_interval]) / self.visual_interval)
            losses.append(sum(self.agent.losses[start:start+self.visual_interval]) / self.visual_interval)

        xticks = np.arange(0, len(self.progress_list), self.interval)

        if progress:
            if len(progresses) > 50:
                plt.xticks(xticks, [str(x) + 'K' for x in xticks // 10])
            plt.axhline(y=(sum(self.progress_list)/len(self.progress_list)), color='b', linestyle='-')
            plt.scatter(range(len(progresses)), progresses, marker='.',alpha=0.3,
                        color=['red' if x == max(progresses) else 'black' for x in progresses])
            plt.annotate(max(progresses), (progresses.index(max(progresses))+5, max(progresses)))
            plt.title(f"Median Progress per {self.visual_interval} episodes")
            plt.show()

        if win_rates:
            if len(progresses) > 50:
                plt.xticks(xticks, [str(x) + 'K' for x in xticks // 10])
                plt.axhline(y=(sum(self.wins_list)/len(self.wins_list)), color='b', linestyle='-')
                plt.axhline(y=(sum(self.wins_list[-100:])/len(self.wins_list[-100:])), color='b', linestyle='--')
            plt.fill_between(range(len(win_rates)), min(win_rates), win_rates, alpha=0.7)
            plt.scatter(win_rates.index(max(win_rates)), max(win_rates), marker='.', color='r')
            plt.annotate(max(win_rates), (win_rates.index(max(win_rates))+5, max(win_rates)))
            plt.title(f"Median Win rate per {self.visual_interval} episodes")
            plt.show()

        if rewards:
            if len(progresses) > 50:
                plt.xticks(xticks, [str(x) + 'K' for x in xticks // 10])
                plt.axhline(y=(sum(self.ep_rewards_list)/len(self.ep_rewards_list)), color='b', linestyle='-')
            plt.scatter(range(len(rewards)), rewards,
                        marker='.', alpha=0.3, color=['red' if x == max(rewards) else 'black' for x in rewards])
            plt.annotate(round(max(rewards),2), (rewards.index(max(rewards))+5, max(rewards)))
            plt.title(f"Median Episode Reward per {self.visual_interval} episodes")
            plt.show()

        if losses:
            if len(progresses) > 50:
                plt.xticks(xticks, [str(x) + 'K' for x in xticks // 10])
            plt.plot(losses)
            plt.title(f"Median Loss per {self.visual_interval} episodes")
            plt.show()

    def save_model(self):

        def save_file(direction, fname, file):
           with open(os.path.join(direction, f'{fname}.pkl'), 'wb') as f:
                pickle.dump(file,f)

        def create_file(path, name):
            file_path = path + '/' + name
            # 파일이 이미 존재하는지 확인
            if not os.path.exists(file_path):
                os.makedirs(file_path)
                print(f"파일 '{file_path}'가 생성되었습니다.")
            else:
                print(f"파일 '{file_path}'는 이미 존재합니다.")

        save_point = {}
        save_point['n_mines'] = self.env.total_mines
        save_point['total_episodes'] = len(self.progress_list)
        save_point['final_model'] = self.agent.model.state_dict()
        save_point['best_model_train'] = self.best_model_train
        save_point['best_model_valid'] = self.best_model_valid
        save_point['best_model_successed'] = self.best_model_successed

        self.save_point = save_point

        create_file(self.f_path, self.name)
        save_file(self.total_path, f'{len(self.progress_list)}epi_train : {self.baseline_train} | valid : {self.baseline_valid} | success : {self.baseline_successed}',save_point)
        print('모델이 저장되었습니다.')



#  if __name__ == "__main__":
#     trainer = Trainer(env, 
#                       agent, 
#                       tester_agent, 
#                       name, 
#                       train_start=True, 
#                       EPISODES = EPISODES,
#                       PRINT_INTERVAL = PRINT_INTERVAL,
#                       TRAIN_RENDER = TRAIN_RENDER,
#                       TRAIN_TIMESTEP = TRAIN_TIMESTEPS[0],
#                       VISUAL_INTERVAL = VISUAL_INTERVAL,
#                       VALID_SAMPLE = VALID_SAMPLE,
#                       VALID_INTERVAL = VALID_INTERVAL)