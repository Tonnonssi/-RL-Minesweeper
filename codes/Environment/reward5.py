import pandas as pd
import numpy as np
import copy
from collections import deque
from IPython.display import display
import pickle
import random

class MinesweeperEnv:
    '''
    This env has 5 rewards : win, lose, progress, guess, and no_progress. 
    '''
    def __init__(self, 
                 map_size, 
                 n_mines, 
                 rewards={'win':1, 'lose':-1, 'progress':0.3, 'guess':-0.3, 'no_progress' : -0.3}, 
                 dones={'win':True, 'lose':True, 'progress':False, 'guess':False, 'no_progress' : False}):
        
        # 지뢰찾기 맵에 대한 기본 정보
        self.map_size = map_size
        self.nrows, self.ncols = map_size
        self.total_tiles = self.nrows*self.ncols # n_tiles에서 변경함
        self.total_mines = n_mines

        # 학습을 위한 정보
        self.rewards = rewards
        self.dones = dones

        # 지뢰찾기 판 생성
        self.board = self.make_init_board()

        # state 생성
        self.state = self.create_state(self.board)

        # 상황 판단을 위한 파라미터
        self.unrevealed = -1.0 / 8.0

    def seed_mines(self):
        actual_board = np.zeros(shape=self.total_tiles, dtype='object')

        # 지뢰 생성
        mine_indices = np.random.choice(self.total_tiles, self.total_mines, replace=False)
        actual_board[mine_indices] = "M"

        # actual board map_size로 복구
        actual_board = actual_board.reshape(self.map_size)

        return actual_board

    def complete_actual_board(self, actual_board):
        padded_actual_board = np.pad(actual_board, pad_width=1, mode='constant', constant_values=0)
        completed_actual_board = actual_board

        for x in range(0, self.nrows):
            for y in range(0, self.ncols):
                if actual_board[x, y] == "M":
                    continue
                else:
                    kernel = padded_actual_board[x:x+3, y:y+3] # padded_actual_board에서의 x,y값은 기존의 +1이라서
                    # kernel[1,1] = 0 _ 논리 상으로는 있는게 맞지만 없어도 문제는 안된다. 중앙이 지뢰일 경우가 없기 때문에
                    completed_actual_board[x, y] = np.sum(kernel == 'M')

        return completed_actual_board

    def make_init_board(self):
        board = np.ones(shape=(2,self.nrows, self.ncols),dtype='object') # (revealed_or_not, game_board)
        actual_board = self.seed_mines()
        actual_board = self.complete_actual_board(actual_board)
        board[1] = actual_board

        return board

    def create_state(self, board):
        revealed_mask = board[0]
        actual_board = copy.deepcopy(board[1])

        # trainable한 형태로 변환
        actual_board[actual_board == "M"] = -2

        masked_state = np.ma.masked_array(actual_board,revealed_mask)
        masked_state = masked_state.filled(-1) # -1은 unrevealed를 의미한다.

        scaled_state = masked_state / 8
        scaled_state = scaled_state.astype(np.float16)

        return scaled_state

    def get_coord(self, action_idx):
        # 선택한 action을 더 가시적이게 나타내기 위해

        x = action_idx // self.ncols
        y = action_idx % self.ncols

        return (x, y)

    def click(self, action_idx):
        # click한 타일을 reveal
        clicked_coord = self.get_coord(action_idx)
        self.board[0][clicked_coord] = 0
        value = self.board[1][clicked_coord]

        unrevealed_mask = self.board[0] # revealed : 0, unrevealed : 1
        actual_board = self.board[1].reshape(1,self.total_tiles)

        # 첫 번째로 선택한 타일은 지뢰가 아니어야 함.
        if (value == 'M') & (np.sum(unrevealed_mask == 0) == 1):
            safe_tile_indices = np.nonzero(actual_board!='M')[1]
            another_move_idx = np.random.choice(safe_tile_indices)
            another_move_coord = self.get_coord(another_move_idx)

            # 지뢰를 이전한다.
            self.board[1][another_move_coord] = 'M'
            self.board[1][clicked_coord] = 0 # 초기화 용

            # 갱신한 내용을 바탕으로 다시 판을 계산한다.
            self.board[1] = self.complete_actual_board(self.board[1])
            value = self.board[1][clicked_coord]

        # 선택한 타일이 0이라면 주변의 타일이 깨진다.
        if value == 0.0:
            self.reveal_neighbors(clicked_coord)

    def reveal_neighbors(self, coord):
        queue = deque([coord])
        seen = set([coord])
        while queue:
            current_coord = queue.popleft()
            x,y = current_coord

            if self.board[1][x,y] == 0:
                for col in range(max(0,y-1), min(y+2, self.ncols)):
                    for row in range(max(0,x-1), min(x+2,self.nrows)):
                        if (row, col) not in seen:
                            seen.add((row, col))
                            queue.append((row, col))

                            self.board[0][row, col] = 0 # 아마 필요없을 것 

    def reset(self):
        # 지뢰찾기 판 생성
        self.board = self.make_init_board()
        # state 생성
        self.state = self.create_state(self.board)

    def step(self, action_idx):
        done = False
        coord = self.get_coord(action_idx)

        current_mask = copy.deepcopy(self.board[0])

        # action에 따라 행동을 수행
        self.click(action_idx)

        # update state
        next_state = self.create_state(self.board)
        self.state = next_state

        # About Reward
        if self.board[1][coord] == 'M':
            reward = self.rewards['lose']
            done = self.dones['lose']

        elif np.sum(self.board[0] == 1) == self.total_mines:
            reward = self.rewards['win']
            done = self.dones['win']

        elif current_mask[coord] == 0: # 이미 깐 타일을 눌렀을 때
            reward = self.rewards['no_progress']
            done = self.dones['no_progress']

        else:
            padded_unrevealed = np.pad(current_mask, pad_width=1, mode='constant', constant_values=1)
            
            if np.sum(padded_unrevealed[coord[0]:coord[0]+3, coord[1]:coord[1]+3] == 1) == 9: # 아무 정보 없이 누른 타일
                reward = self.rewards['guess']
                done = self.dones['guess']

            else:
                reward = self.rewards['progress']
                done = self.dones['progress']

        return self.state, reward, done

    def render(self, state):
        # 원래 값으로 복구한 뒤 시각화한다.
        state = (state * 8.0).astype(np.int8)
        state = state.astype(object)
        state[state == -1] = '.'
        state[state == -2] = 'M'
        state_df = pd.DataFrame(state.reshape((self.map_size)))

        display(state_df.style.applymap(self.color_state))
        print(" ")

    def color_state(self, value):
        if value == '.':
            color = 'white'
        elif value == 0:
            color = 'slategrey'
        elif value == 1:
            color = 'blue'
        elif value == 2:
            color = 'green'
        elif value == 3:
            color = 'red'
        elif value == 4:
            color = 'midnightblue'
        elif value == 5:
            color = 'brown'
        elif value == 6:
            color = 'aquamarine'
        elif value == 7:
            color = 'black'
        elif value == 8:
            color = 'silver'
        else:
            color = 'magenta'

        return f'color: {color}'
    
class LimitedMinesweeperEnv(MinesweeperEnv):
    def __init__(self, map_size, n_mines, total_boards=None, train=True):
        super().__init__(map_size, n_mines)

        self.train = train

        if total_boards is None:
            with open("/content/drive/MyDrive/Minesweeper [RL]/dataset/easy1000boards.pkl","rb") as f:
                self.total_boards = pickle.load(f)
        else:
            self.total_boards = total_boards

        self.n_boards = len(self.total_boards)

        if train:
            self.board = self.total_boards[0]
        else:
            self.board_iteration = iter(self.total_boards)
            self.board = next(self.board_iteration)

    def reset(self):
        self.n_clicks = 0
        self.n_progress = 0

        if self.train:
            self.board = random.choice(self.total_boards)
            self.board[0] = np.ones(shape=self.map_size) # board가 수정되기 때문에 초기화해줘야 한다.
        else:
            try:
                self.board = next(self.board_iteration)
            except StopIteration:
                # Optionally: Reinitialize the iterator if you want to cycle through the boards
                self.board_iteration = iter(self.total_boards)
                self.board = next(self.board_iteration)

        self.state = self.create_state(self.board)

