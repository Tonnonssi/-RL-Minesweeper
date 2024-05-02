import sys
sys.path.append('/content/drive/My Drive/Minesweeper [RL]/codes')

from Agent.DQNAgentWithRules import *

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