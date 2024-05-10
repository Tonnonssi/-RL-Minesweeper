import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

class PerformTester:
    def __init__(self, agent, env, n_episode, model_state):
        # 기본 파라미터
        self.agent = agent
        self.env = env
        self.n_episode = n_episode
        self.model_state = model_state

        # test model & stack replay memory
        self.replay_memory = []
        self.test_model(n_episode, model_state)

        # memory_df
        self.memory_df = pd.DataFrame(self.replay_memory, columns=['episode', 'board', 'current_state', 'action', 'Qtable', 'reward', 'next_state', 'done'])

        # make df of lost_game, won_game
        self.lost_game_epi = self.memory_df[self.memory_df['reward'] == self.env.rewards['lose']]['episode'].tolist()
        self.won_game_epi = self.memory_df[self.memory_df['reward'] == self.env.rewards['win']]['episode'].tolist()

        self.lost_game = self.memory_df[self.memory_df['episode'].isin(self.lost_game_epi)]
        self.won_game = self.memory_df[self.memory_df['episode'].isin(self.won_game_epi)]

        self.lost_game_per_epi = iter(self.lost_game.groupby('episode'))
        self.won_game_per_epi = iter(self.won_game.groupby('episode'))

        self.lost_game_done = self.lost_game[self.lost_game['done'] == True]
        self.won_game_done = self.won_game[self.won_game['done'] == True]

        # 18개 이상 까졌는데 진 케이스
        self.lost_more18_mask = self.lost_game['current_state'].apply(lambda x: np.sum(x != self.env.unrevealed) > 18)
        self.lost_more18_epi = self.lost_game[self.lost_more18_mask]['episode'].tolist()

        self.lost_more18_game = self.memory_df[self.memory_df['episode'].isin(self.lost_more18_epi)]
        self.lost_more18_per_epi = iter(self.lost_more18_game.groupby('episode'))
        self.lost_more18_done = self.lost_more18_game[self.lost_more18_game['done'] == True]

        self.lost_more18_percent = round(sum(self.lost_more18_mask) / len(self.lost_more18_mask),3)


        # palette for visualize
        minesweeper_cmap = ['#FF00FF', '#FFFFFF', '#6A5ACD', '#0000FF', '#008000', '#FF0000',
                    '#191970', '#A52A2A', '#7FFFD4', '#000000', '#C0C0C0']
        self.minesweeper_palette = sns.color_palette(minesweeper_cmap)

    def test_model(self, n_episode, model_state):
        print("Test Started.")

        progress_list, wins_list, ep_rewards = [], [], []

        self.agent.epsilon = 0.0
        self.agent.model.load_state_dict(model_state)
        self.agent.target_model.load_state_dict(model_state)

        for episode in range(n_episode):
            self.env.reset()

            done = False
            n_clicks = 0
            episode_reward = 0

            while not done:
                current_state = self.env.state
                action = self.agent.get_action(current_state)
                next_state, reward, done = self.env.step(action)

                episode_reward += reward
                n_clicks += 1

                if (current_state == next_state).all(): # 같은 곳을 계속 누르는 상황을 탈출시키는 ShutDown Code
                    done = True

                self.replay_memory.append((episode, self.env.board, current_state, action, self.agent.total_action.reshape(self.env.map_size).numpy(), reward, next_state, done))

            progress_list.append(n_clicks)
            ep_rewards.append(episode_reward)
            wins_list.append(reward == self.env.rewards['win'])

        print(f"Test [n: {n_episode}], Median progress: {np.median(progress_list):.2f}, Median reward: {np.median(ep_rewards):.2f}, Win rate : {np.sum(wins_list)/len(wins_list)}")

    def visualize_single_step(self, df):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

        sns.heatmap(df['current_state'].reshape(self.env.map_size)*8,
                    cmap=self.minesweeper_palette,
                    annot=True,
                    linewidth=.5,
                    fmt=".0f",
                    vmin=-2,
                    vmax=8,
                    ax=ax1)

        sns.heatmap(df['next_state'].reshape(self.env.map_size)*8,
                    cmap=self.minesweeper_palette,
                    annot=True,
                    linewidth=.5,
                    fmt=".0f",
                    vmin=-2,
                    vmax=8,
                    ax=ax2)

        # scaling Q-table
        min_value = np.min(df['Qtable'])
        max_value = np.max(df['Qtable'])
        q_table = (df['Qtable'] - min_value) / (max_value - min_value)

        sns.heatmap(q_table,
                    cmap='PuBuGn',
                    annot=True,
                    linewidth=.5,
                    fmt=".2f",
                    ax=ax3)

        # Q-table에 지뢰 위치 표기
        board = df['board'][1]
        mine_coords = np.where(board == 'M')

        ax3.scatter(mine_coords[1] + 0.5, mine_coords[0] + 0.5, marker='D', s=250, color='orange')
        ax2.scatter(mine_coords[1] + 0.5, mine_coords[0] + 0.5, marker='D', s=250, color='#FF00FF')

        ax1.set_title('Current State')
        ax2.set_title('Next State')
        ax3.set_title('Q-table')
        plt.tight_layout()
        plt.show()

    def replay_single_episode(self, epi_df):
        # df는 episode를 기준으로 나눠져 있어야 한다.
        epi_df.apply(lambda x: self.render(x['next_state'], x['action']), axis=1)

    def render(self, state, action):
        # 원래 값으로 복구한 뒤 시각화한다.

        def fill_clicked_coord(x, color):
            color = f'background-color:{color}'
            return color

        def get_coord(action):
            return (action // self.env.nrows, action % self.env.nrows)

        coord = get_coord(action)

        state = (state * 8.0).astype(np.int8)
        state = state.astype(object)
        state[state == -1] = '.'
        state[state == -2] = 'M'
        state_df = pd.DataFrame(state.reshape((self.env.map_size)))

        print(f"coord {coord} is clicked.")

        # 누른 타일 시각화
        styled_df = state_df.style
        idx = pd.IndexSlice
        styled_df = styled_df.applymap(fill_clicked_coord, color='#feea67', subset=idx[coord[0],coord[1]])
        styled_df = styled_df.applymap(self.env.color_state)

        display(styled_df)
        print(" ")