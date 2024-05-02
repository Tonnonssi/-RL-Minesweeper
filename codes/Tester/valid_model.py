import numpy as np

def valid_model(env, agent, episode, epoch, model_state):

    progress_list, wins_list, ep_rewards = [], [], []

    agent.epsilon = 0.0
    agent.model.load_state_dict(model_state)
    agent.target_model.load_state_dict(model_state)

    for _ in range(epoch):
        env.reset()

        done = False
        n_clicks = 0
        episode_reward = 0

        while not done:
            current_state = env.state

            action = agent.get_action(current_state)
            _, reward, done = env.step(action)

            episode_reward += reward
            n_clicks += 1

        progress_list.append(n_clicks)
        ep_rewards.append(episode_reward)

        if reward == env.rewards['win']:
            wins_list.append(1)
        else:
            wins_list.append(0)

    print(f"Valid [{episode+1}]n:{epoch}, Median progress: {np.median(progress_list):.2f}, Median reward: {np.median(ep_rewards):.2f}, Win rate : {np.sum(wins_list)/len(wins_list)}")

    return np.sum(wins_list)/len(wins_list) # 승률을 반환한다. 