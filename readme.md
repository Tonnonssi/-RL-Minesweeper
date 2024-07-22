## [RL] MineSweeper
<<<<<<< HEAD
## 00. Overview 
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Visual Studio](https://img.shields.io/badge/Visual%20Studio-5C2D91.svg?style=for-the-badge&logo=visual-studio&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

이 프로젝트는 DQN / DDQN 강화학습 방법론을 사용해 지뢰찾기를 풀어내는 것을 목표로 한다. 
### About Minesweeper 
지뢰찾기는 맵에 깔려있는 모든 지뢰를 찾는 게임이다. 여기서 '찾는다'는 것은 지뢰라고 판단되는 타일을 건들이지 않는 것이기 때문에, 지뢰찾기의 승리조건은 지뢰가 아닌 나머지 모든 타일을 다 까는 것과 같다. 이 맥락에서 일반적으로 지뢰찾기 게임 내 존재하는 깃발 기능은 필수가 아니다. 깃발은 찾아낸 지뢰를 표시하기 위해 있는 기능이다. 승리조건에 미루어 비춰 볼 때, 깃발은 플레이어의 편의성을 위해 존재하는 기능일 뿐 필수요건은 아니다. 따라서 본 프로젝트에서는 깃발 기능을 구현하지 않고 진행했다. (추후 업데이트 예정)  

지뢰찾기의 난이도는 초급 중급 고급으로 나누어져 있다.  

| Level | Height | Width | Num of Mine | density | 
| - | - | - | - | - | 
| 초급 | 9 | 9 | 10 | 12.3% | 
| 중급 | 16 | 16 | 40 | 15.6% | 
| 고급 | 16 | 30 | 99 | 20.6% | 

구현된 환경은 커스텀 포함 모든 난이도로 학습이 가능하지만, 본 프로젝트는 초급(9*9, 지뢰 10개)을 기준으로 진행했다.  

### RL 
DQN / DDQN 방법론을 이용했으며, 구체적인 내용 및 구현은 Ch.03에 기재되어 있다. DeepMind Gym 스타일을 따랐으며, `step()`, `reset()`, `render()`, `action_space`가 구현되어 있다.  

### 학습 환경 
코랩에서 학습을 진행했다. T4 GPU / L4 GPU / A100 GPU를 사용했고, A100을 기준으로 20만 에피소드(valid 포함) 6시간 25분이 소요됐다. 

## Contents
- 목차

## 01. Environoment
### Package
``` 
0.0.0
```

### State 
#### 2D type
9*9(easy의 맵 규격)의 arr에 주변부 지뢰의 개수를 센 타일의 수(0-8)를 그대로 반영하고, 아직 까지지 않은 타일을 -1, 지뢰를 -2로 표현했다. 최댓값인 8로 전체 state arr를 나누어 state를 정규화했다. 정규화를 하지 않았을 때엔 학습이 거의 되지 않았지만, 정규화 이후에 승률이 극적으로 상승했다. 
![Alt text](image-2.png)

#### 3D type


### Reward Design
| Reward | Description & Purpose | Weight | Done | 
| - | - | - | - | 
| Win | 지뢰가 아닌 모든 타일을 깐 경우 | 1 | True |
| Lose | 깐 타일이 지뢰인 경우 | -1 |True |
| Guess | 주변부가 까져 있지 않지만, 지뢰가 아닌 타일을 깐 경우 | 0.1 | False |
| Progess | 주변부에 까진 타일이 있고, 지뢰가 아닌 타일을 깐 경우 | 0.3 | False |
| No Progess | 이미 누른 타일을 또 누른 경우 | -1 | T/F |  

- 보상 디자인의 변화는 Ch.02 Agent / 05. Result 에서 확인할 수 있다. 


### Improve code efficiency
n만 번 단위로 에피소드를 진행하다보니, 환경 코드의 시간 효율이 중요했다. 시간 효율을 높이기 위해 0을 눌렀을 때 연쇄로 터지는 코드, 주변 지뢰의 개수를 세는 코드를 리팩토링했다. 

#### 1. 0 bombing chain
기존 코드는 재귀형으로 구현했지만, 무한 루프에 빠지는 문제가 있었다. 무한 루프 문제를 해결하고 속도를 향상시키기 위해서 BFS 알고리즘을 이용해 코드를 수정했다. 

#### 2. count neighbor mine's num
기존 코드는 특정 타일을 기준으로 주변 타일을 탐색하는 방식을 이중 for문으로 구현했다. for문을 사용을 최소화 하고자, 겉에 0 패딩을 두른 후 3*3로 arr를 순서대로 탐색하며 M의 개수를 세는 방식으로 코드를 수정했다.  
#### 3. cython  
to be continue...

## 02. Agent 
### Action with Rule vs only with Rewards
구현 초기에는 이미 까진 타일의 Q값을 전체 Q 값 중 최소값으로 죽여 강제로 선택을 하지 못하게 만들었다. 이미 까진 타일을 또 누르는 것은 에이전트를 패배로 이끌지는 않지만, 갇히게 되면 무한 에피소드를 만든다. 따라서 이 값을 강제로 죽이는 것이 학습에 있어 일종의 안내판이 될 것이라 기대했지만, 최대 56% 평균 44%를 웃도는 성능에 그쳤다. 여러 수정 이후에도 성능이 나아지지 않자, 이미 까진 타일에 음수 보상을 줘 누르지 않도록 행동을 유도하는 방식을 적용했고, 80%p 이상의 성능 향상이 이뤄졌다.  

![Alt text](image-7.png)

이미 까진 Q 값을 강제로 죽이는 방식이 DQN의 부담을 덜어줄 것이라 생각했지만, 결과는 정 반대였다. 이를 계기로 강제로 값을 죽이는 방식이 어떤 문제점과 한계가 있는지 고민했다.  

1. 이미 까진 타일을 또 누르는 행동은 게임 클리어를 막는 주요 고려 대상이다. 구현 초반에는 이 행동이 승리나 패배에 기여하지 않기 때문에 그다지 의미가 없는 행동이라 생각했다. 하지만 지뢰찾기는 지뢰를 제외한 모든 타일을 까는 게임이다. 즉, 이미 까진 타일을 또 누르는 행동은 게임의 승리에 다가가도록 만들지 못하기 때문에 게임 승리를 방해하는 요소다. 따라서 이 행동을 강제로 제한하는 것은, 지뢰가 아닌 모든 타일을 전부 다 까도록 제대로 유도하지 못하기 때문에 오히려 환경의 복잡도가 증가한다.  

3. 행동을 제한하는 것이 DQN의 모든 부분에 적용되지 않는다. 이미 까진 타일을 누르지 못하게 하는 것은 행동을 정하는 매소드인 `get_action(state)`에만 적용했다. 하지만 이 상황에서는 리플레이 메모리에서 state-action 페어를 꺼내 예측값을 만들 때는 문제가 없지만, next_state를 받아 다음 타겟값을 만드는 타겟 신경망에서 문제가 생긴다. 타겟 신경망을 통해 얻은 Q 값들 중 최대값은 타겟값이 되는데, 이때 최대값을 갖는 행동은 이미 까진 타일을 누르는 행동일 수 있다. 이 값은 next_state가 줄 수 있는 최대의 가치이지만, 리플레이 메모리의 `state-action-reward`에는 이미 까진 곳을 누르는 사건이 존재하지 않기 때문에 제대로 된 값이 나오도록 학습된 신경망을 사용할 수 없다. 따라서 이 값은 제대로 근사될 수 없기에 신경망의 한계를 야기시킨다. 

 따라서 강제로 행동을 하지 못하게 막는 것보단, 에이전트가 누르지 않도록 유도하는 방법을 찾는 것이 오히려 환경의 복잡도를 낮출 수 있다. 

## 03. DQN
### Net 
CNN은 공간상의 정보를 인식할 수 있기 때문에 이미지 인식에 유리하다. 지뢰찾기의 state 또한 수와 주변부 사이의 패턴이 중요하기 때문에 CNN 신경망을 이용했다. 패딩과 합성곱 층만을 이용하는 두 가지 아이디어에서 성능 향상이 이뤄졌다. 

#### 1. CNN with Padding
합성곱 신경망에서 패딩은 이미지 크기 손실을 막고, 가장자리를 더 잘 인식하게 만들어준다. 가장자리 값이 중앙이나 다른 영역에 비해 중요도가 낮을 수 있는 일반적인 사진과는 달리, 지뢰찾기에서 가장자리는 중앙과 동일하게 중요한 정보값을 갖고 있다. 커넬이 state의 모든 타일을 동일하게 탐색하여 정보를 얻어가게 만들기 위해 첫번째 합성곱 layer에서는 2단의 패딩을, 나머지 layer에서는 1단의 패딩을 적용했다. 또한 이미지의 크기가 유지되었기 때문에, 9*9라는 작은 state에서 4층 이상의 다층 신경망을 적용할 수 있었다. 그 결과 패딩이 없는 모델에서는 학습이 거의 진행되지 않았던 것에 반해, 성공률이 높아지며 학습이 되는 양상을 보였다. 

#### 2. CNN only with conv layer
[@ryanbaldini](https://github.com/ryanbaldini/MineSweeperNeuralNet)의 모델을 통해 합성곱 신경망만을 사용하는 아이디어를 얻었다. 추가적인 합성곱 신경망을 쌓지 않고 전연결 신경망을 삭제하는 것만으로 100%p 이상의 성능 상승을 보였다. 명확한 원인을 규정할 수는 없었지만 합성곱 층에서 뽑아낼 수 있는 정보들이 전연결 신경망을 거치며 오히려 꼬여 영양가를 잃는 것이 아닐까라는 가설을 세웠다. 
![Alt text](image-3.png)

### Replay Memory 
추측 가능할 만큼 타일이 까지지 않는다면 사람이 게임을 플레이해도 찍을 수 밖에 없다. 때문에 추측이 불가능할 정도로 적게 타일이 까여진 state는 학습에 있어 주요한 데이터가 아니라 판단했다. replay memory에 저장할 타일의 기준을 세우기 위해 직접 지뢰찾기를 플레이하며 추측을 통해 풀 수 있다 판단한 시점에 까여진 타일의 개수를 세었다. 그 결과 30개의 표본에서 평균 18개라는 수치를 얻을 수 있었다. 이를 바탕으로 에피소드를 진행하며 18개 미만으로 까여진 state는 replay memory에 저장하지 않는 방식으로 replay memory를 수정했다. 그 결과 초반 학습이 유의미하게 빨라지는 것을 확인할 수 있었다. 
![Alt text](image.png)  

(+) 이 방식은 초반 수렴을 가속화하기 때문에, 특정 신경망 및 보상 체계가 잘 작동하는지 확인하는데 용이하다. 다만 의도적으로 일부 state를 제외하는 것은 모든 환경을 온전히 담지 못한다. 이런 한계를 인지하고 보상 체계 및 신경망이 안정화된 후 최종적으로 가장 높은 승률을 지닌 모델을 찾을 때는 기능을 사용하지 않았다. 

### vector type / scalar type Q-learning
![Alt text](image-4.png)  
- **DQN 알고리즘** 
![Alt text](image-5.png)

DQN 알고리즘을 두 가지 버전으로 구현했다. 먼저 vector type은 [@pythonprogramming.net](https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/)의 코드를 참고했으며, 텐서플로우에서 파이토치로 변환할 때 불필요한 구현부를 최소화시켜 속도를 향상시켰다. 이 코드는 state를 신경망에 넣었을 때 나오는 전체 Q 값을 예측값으로 사용한다. 타겟값 또한 예측값의 Q값을 복사한 후, 타겟 신경망의 최대 Q 값을 기존의 action이 있는 위치에 넣는 방식으로 DQN 알고리즘을 구현한다. scalar 타입은 [pytorch DQN 공식문서](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)를 참고해 구현했으며, 예측값은 신경망에서 구한 Q 값들 중 a에 해당하는 값이고 타겟 값은 타겟 신경망에서 얻은 Q 값 중 가장 큰 값이다. 

벡터 타입의 DQN과 스칼라 타입의 DQN의 차이점은 loss값이다. MSE 기준으로 생각했을 때, 벡터 타입의 DQN은 하나의 숫자만 다르고 나머지가 전부 동일한 두 벡터를 비교하기 때문에 스칼라와 예측값과 타겟값 사이의 오차제곱과는 동일하다. 하지만 MSE loss는 오차제곱 값의 평균을 다루기 때문에, 벡터 타입의 DQN이 스칼라 타입보다 loss값이 작다.  

이론적으로는 스칼라 타입이 더 적합한 방법이라 생각하지만, 벡터 타입 또한 큰 문제는 없다 판단해 두 방식을 학습에 전부 다 이용해보았다. 그 결과, 전부 같은 조건에서 학습을 진행했을 때 스칼라 타입이 벡터 타입에 비해 이미 까진 타일을 여러 번 눌러 전체 에피소드의 스텝 수가 증가하고, 총 리워드가 감소하는 모습이 더 강하게 오래 동안 발생했다. 
![Alt text](image-6.png)
이를 통해 낮은 loss값이 더 빠른 수렴에 도움을 줄 수 있다 생각해 주로 초기 수렴이 빠른 벡터 타입을 이용해 학습을 진행했다. 
## 04. Train / Valid / Test
### Track training : mean, median, mode 
한 에피소드 당 몇 step을 갔는지, 이겼는지 졌는지, 총 보상이 얼마인지를 학습 도중 학습이 되고 있나 확인하는 용으로 사용했으며, 100 에피소드마다 출력되었다. 이때 100 에피소드를 대표할 수 있는 값을 사용하는 것이 중요했고, 평균 중앙값 최빈값 중에서 중앙값을 선택했다. 중앙값은 평균과 달리 outlier에 영향을 덜 받는다. 지뢰찾기는 랜덤성이 가미되는 게임으로 잘 학습된 모델이라도 추측이 불가능한 초반 판에서는 찍기를 선택해야하며, 난이도가 올라갈수록 게임 도중 추론만으로는 해결할 수 없는 구간이 발생한다. 특히 진행한 초급 단계에서는 초반 판에서의 찍기가 가장 주된 랜덤성의 원인이었기 때문에 평균을 사용하기엔 극단값이 많이 포함될 위험성이 있었고, 이를 방지하고자 중앙값을 사용했다. 같은 맥락에서 최빈값은 나쁜 지표는 아니었지만, 전체적인 경향을 보기에는 중앙값이 더 좋다고 판단했다. 

### Importance of Validation
학습 도중의 지표들은 전체적인 학습 진척을 보기에는 좋지만, 가장 승률이 높은 모델을 찾기에는 허점이 존재한다. 학습은 100단위의 에피소드의 결과만을 알려주기 때문에, 학습 중 가장 높은 승률의 모델을 저장한다는 것은, 가장 승률이 높았던 100개의 에피소드 구간의 맨 마지막 모델이다. 매 timestep마다 신경망이 업데이트되고 있는 상황에서 학습 중 저장되는 가장 마지막 모델이 전체 학습에서 존재한 모델 중 최고 성능을 지닌다고 이야기하기는 어렵다.  
validation은 이 문제를 해결할 수 있는 가장 간편한 방법이다. 특정 시점마다 모델을 고정시킨 채 랜덤한 n개의 게임을 실행시켜 승률을 구해 validation을 진행할 수 있다. 모든 모델에 대해 진행하는 것이 가장 정확하겠지만 비용대비 효율적이지 못하다. 학습 도중의 승률이 갱신되었을 때, valid에서의 승률이 갱신되었을 때마다 10 에피소드 텀을 두고 총 10번의 valid를 실행했다. 또한 더 많은 모델을 수집하기 위해, 가장 마지막에 승리한 모델, 학습의 맨 마지막 모델, valid에서 가장 높은 승률을 지닌 모델, 학습에서 가장 높은 승률을 지닌 모델 총 4개를 저장해 test에 사용했다. 

### Proper Sample Size for Valid
추가하ㅐ야해

### Visualize in Test
- test에서 모델의 성능을 판단하고, 각 에피소드의 state-action-Q table을 확인했다. 
![Alt text](image-8.png)
순서대로 현재 상태, 행동으로 인한 다음 상태, 선택에 영향을 주는 Q-table이다. 마젠타 색 마름모로 맵 상에서 지뢰가 어디 숨겨져 있는지를 표현했고, Q-table의 값은 실제 신경망이 뱉은 값이 아닌, min-max로 스케일링한 결과물이다. 일반 Q-table로 보면 행동의 우선순위가 잘 보이지 않았기 때문에 0-1 사이의 수로 들어오도록 스케일링했다. 

- env.render에 트랙커 기능을 더했다. 
![Alt text](image-9.png)
기존 env.render은 누른 타일을 텍스트로만 전달해, 누른 타일을 즉시 파악할 수 없었다. 1차적으로는 좌표로 변환해 문제를 해결했지만, 좌표를 보고 판에 적용해야 했기 떄문에 효과적이지 않았다. 따라서 누른 좌표를 tracker로 사용해 색을 칠하는 방식으로 문제를 해결했다.

## 05. Trials
### 이미 열린 타일을 누르는 행동 
- 이미 열린 타일을 누르는 행동 : `no_progress`  

강화학습에서 지뢰찾기를 풀 때 가장 문제가 되는 행동은 이미 열린 타일을 또 누르는 행동이다. 학습 단계는 탐험률이 존재하기 때문에 이미 열린 타일을 누르다가도 다른 행동으로 탈출할 수 있다. 하지만 valid나 test에서는 탐험률이 존재하지 않기 때문에 무한 에피소드에 빠지게 된다. 이 문제를 해결하기 위해 시도한 방법론들이다.  

#### 1. valid / test에서 이미 열린 타일을 누르면 에피소드 강제 종료 
이미 까진 타일을 강제로 선택하지 못하게 만들지 않으면, valid에서 이미 누른 타일을 또 누르는 행동은 필연적으로 발생한다. 내 valid는 학습 도중 최고 승률이 나올 때마다 실행되기 때문에 초반부터 valid가 돌아가게 된다. 이 시기는 입실론에 의해, 운에 의해 승리에 도달하는 경우가 대부분이기에 환경의 구조를 제대로 이해하지 못한다. 따라서 실제 보상 회로와 관계없이, valid나 test에는 이미 까져있는 타일을 누르면 강제로 에피소드를 종료해 무한 에피소드 문제를 방지했다. 그 결과 학습에서는 랜덤한 행동을 통해 탈출할 수 있었던 에피소드가 강제로 종료되어, 학습과 valid/test 사이에 20% 가량의 성능 차이가 존재했다. 

#### 2. 이미 까진 타일의 Q 값을 강제로 죽여 선택하지 못하게 만듦 
- Ch2 에 기술 
#### 3. no progress : done = True 
이미 누른 타일을 또 누르면 에피소드를 강제로 종료시켰다. 이 방식은 done=False인 방법론들보다 valid/test와 학습의 win rate 격차가 적었다. done의 T/F는 학습 양상에서 차이를 보였다. 같은 것을 눌러도 학습이 종료되다보니 done이 False일 때보다 에피소드의 길이가 절대적으로 짧았고, 그 결과 리플레이 메모리에 차는 데이터 양이 적었다. done이 False인 애들은 5만 에피소드에서 이미 50% 이상의 승률을 보였지만, done이 True인 애들은 5% 정도의 승률을 보였다. 하지만 승률이 안 나올 뿐 에피소드의 총 보상과 step 수는 빠르게 증가했고, 10만 에피소드부터는 done=False인 애들의 성능을 따라잡았다. 60만까지 돌렸을 때 test의 결과는 평균 76%로, done=False인 애들보다 2% 가량 높은 승률을 보였다. 
- Train : 87%, Valid : 80.6%, Test : 78.7% / 74.4% (max, min)


#### 4. Reward : -0.5 --> -1 / 0.9
이미 까진 타일을 또 누르는 행동은 지뢰를 제외한 모든 타일을 까야하는 지뢰찾기의 환경 입장에서 게임의 승리에 다가가지 못하게 만드는 부정적인 행동이다. 처음 no progress의 리워드를 구성할 때엔 이 행동이 단순히 게임을 진전시키지 못할 뿐, 게임의 승패에 직접적으로 관여하지 않는다고 생각해 -0.5(패배시의 절반) 크기의 리워드를 부여했다. 하지만 죽지않고 타일을 까는 행동이 좋고, 그렇지 못한 행동이 전부 다 나쁜 행동이라 지뢰찾기를 단순화시키면 이 행동은 지뢰를 밟는 것만큼이나 나쁜 행동이다. 띠리서 이 때의 보상을 실패했을 때와 동일한 -1을 값을 주되, 지뢰를 밟았을 때와 달리 에피소드가 끝나지 않게 했다. 

### 주변 타일이 까지지 않은 타일을 선택하는 행동
- 주변 타일이 까지지 않은 타일을 선택하는 행동 : `Guess`  

사람이 지뢰찾기를 플레이할 때는 이미 까진 주변부 타일을 통해, 특정 타일이 지뢰일지 아닐지 판단한다. 하지만 강화학습의 에이전트는 지뢰찾기의 룰을 알지 못하며, 주변부 타일을 통해 값을 추론해야 한다는 로직을 알지 못한다. 따라서 나는 주변부 타일이 까지지 않은 타일을 누르는 행동을 Guess라고 정의하고, 일반 진전과는 다른 보상을 주었다.   

#### 초기. 음의 보상을 줌
주변부의 정보가 없는데 타일을 까는 것은 지뢰찾기의 룰을 배우는데 부정적인 행동이라 생각해 진전의 반대가 되는 음의 보상(-0.3)을 주었다.  
- Train : 87%, Valid : 80.6%, Test : 78.7% / 74.4% (max, min) 수정 필요

#### 중기. Guess == Progress  
no progress 단에서 이야기했듯이, 죽지 않고 타일을 까는 행동이 좋고, 그렇지 않은 행동이 전부 다 나쁜 행동이라 지뢰찾기를 단순화시키면 Guess는 죽지 않고 타일을 까기에 좋은 행동이다. 따라서 Guess에 음의 보상을 주는 것은 논리적으로 이상하다 판단해 가장 단순하게 Progress와 구분하지 않고 동일한 양의 보상(0.3)을 주었다.  
- Train : 87%, Valid : 80.6%, Test : 78.7% / 74.4% (max, min) 수정 필요

#### 마무리. Progress보단 적은 양의 보상 
Guess 없이도 80%가 넘는 승률을 보여주었지만, Guess에 Progress보다 적은 양의 보상(0.1)을 주었을 때의 양상이 궁금해 테스트해보았다. 
- Train : 87%, Valid : 80.6%, Test : 78.7% / 74.4% (max, min) 수정 필요  

로 실질적으로 보상을 동일하게 했을 때와 성능에는 큰 차이가 없음을 알 수 있었다. 
### 주요 파라미터
강화학습에서 사용하는 여러 파라미터 중, learning rate가 학습에 가장 큰 영향을 미쳤다. 동일한 상황에서도 초기 lr이 너무 크거나 작으면 학습이 제대로 진행되지 않았다. 또한 계속 같은 크기의 lr로 학습을 시키는 것도 학습의 한계를 초래했다. 


### 06. Result 
### 찰떡궁합 
9*9 게임판에 10개의 지뢰가 숨어져 있는 easy를 기준으로 최대 승률 92%, 평균 승률 78%의 모델을 만들었다. 
### visualize 
### baseline : Rule Base
### Side Story
첫 선택에서 주변부만 깜 
![Alt text](image-16.png)

## Reference

=======
>>>>>>> b258451f028e3099e0bf9dcc2d6763bafc1b08bd
