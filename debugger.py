import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

env = gym.make('MountainCar-v0') #, render_mode="human")

# Hyperparamters
EPISODES = 5000
DISCOUNT = 0.95
EPISODE_DISPLAY = 500
LEARNING_RATE = 0.25
EPSILON = 0.5
EPSILON_DECREMENTER = EPSILON/(EPISODES//4)

position_max = env.observation_space.high[0]  # 0.6
position_min = env.observation_space.low[0]  # -1.2
velocity_max = env.observation_space.high[1]  # 0.07
velocity_min = env.observation_space.low[1]  # -0.07

bucket_size = 50

print(position_max, position_min, velocity_max, velocity_min)

position_window = (position_max - position_min) / bucket_size
velocity_window = (velocity_max - velocity_min) / bucket_size

Q_TABLE = np.random.randn(
    bucket_size, bucket_size, env.action_space.n)  # 50*50*3

# For states
ep_rewards = []
ep_rewards_table = {'ep': [], 'avg': [], 'min': [], 'max': []}

def discretised_state(state):
    discrete_state = np.array([0, 0])  # Initialised discrete array
    discrete_state[0] = (state[0] - position_min) // position_window
    discrete_state[0] = min(bucket_size-1, max(0, discrete_state[0]))

    discrete_state[1] = (state[1] - velocity_min) // velocity_window
    discrete_state[1] = min(bucket_size-1, max(0, discrete_state[1]))

    return tuple(discrete_state.astype(np.int32))

def train(render=False, algorithm="sarsa"):
    epsilon = EPSILON
    for episode in range(EPISODES):
        episode_reward = 0
        done = False
        truncated = False

        if episode % EPISODE_DISPLAY == 0:
            render_state = True
        else:
            render_state = False

        observation, _info = env.reset()
        curr_discrete_state = discretised_state(observation)

        if np.random.random() > epsilon:
            action = np.argmax(Q_TABLE[curr_discrete_state])  # exploit
        else:
            action = np.random.randint(0, env.action_space.n)  # explore

        while not done and not truncated:  # You must add `not truncated` otherwise the reward can be wrong
            new_state, reward, done, truncated, _info = env.step(action)
            new_discrete_state = discretised_state(new_state)

            if np.random.random() > epsilon:
                new_action = np.argmax(Q_TABLE[new_discrete_state])
            else:
                new_action = np.random.randint(0, env.action_space.n)

            if render and render_state:
                env.render()
                
            if not done:
                current_q = Q_TABLE[curr_discrete_state+(action,)]
                next_q = 0
                if algorithm == "sarsa":
                    next_q = Q_TABLE[new_discrete_state+(new_action,)]
                else:
                    next_q = np.max(Q_TABLE[new_discrete_state])
                Q_TABLE[curr_discrete_state+(action,)] = current_q + \
                    LEARNING_RATE*(reward + DISCOUNT*next_q - current_q)
            elif new_state[0] >= env.goal_position: # 这一行有多大影响？
                Q_TABLE[curr_discrete_state + (action,)] = 0

            curr_discrete_state = new_discrete_state
            action = new_action

            episode_reward += reward

        ep_rewards.append(episode_reward)

        epsilon -= EPSILON_DECREMENTER

        if not episode % EPISODE_DISPLAY:
            avg_reward = sum(ep_rewards[-EPISODE_DISPLAY:]) / \
                len(ep_rewards[-EPISODE_DISPLAY:])
            ep_rewards_table['ep'].append(episode)
            ep_rewards_table['avg'].append(avg_reward)
            ep_rewards_table['min'].append(min(ep_rewards[-EPISODE_DISPLAY:]))
            ep_rewards_table['max'].append(max(ep_rewards[-EPISODE_DISPLAY:]))
            print(
                f"Episode:{episode} avg:{avg_reward} min:{min(ep_rewards[-EPISODE_DISPLAY:])} max:{max(ep_rewards[-EPISODE_DISPLAY:])}")

    env.close()

    plt.plot(ep_rewards_table['ep'], ep_rewards_table['avg'], label="avg")
    plt.plot(ep_rewards_table['ep'], ep_rewards_table['min'], label="min")
    plt.plot(ep_rewards_table['ep'], ep_rewards_table['max'], label="max")
    plt.legend(loc=4) #bottom right
    plt.title('Mountain Car' + algorithm)
    plt.ylabel('Average reward/Episode')
    plt.xlabel('Episodes')
    plt.show()


def play():
    env = gym.make('MountainCar-v0', render_mode="human")
    observation, _info = env.reset(seed=0) # 重置游戏环境，开始新回合。设置随机数种子,只是为了让结果可以精确复现,一般情况下可删去
    epi = 0
    while True: # 不断循环，直到回合结束
        epi += 1
        env.render() # 显示图形界面，图形界面可以用 env.close() 语句关闭
        
        curr_discrete_state = discretised_state(observation)
        action = np.argmax(Q_TABLE[curr_discrete_state])

        next_observation, reward, terminated, truncated, _info = env.step(action) # 执行动作
        print("#", epi, ", action = ", action, ", reward = ", reward)
        if terminated or truncated: # 回合结束，跳出循环
            break
        observation = next_observation
    return -epi # 返回回合总奖励

train()
print('回合奖励 = {}'.format(play()))
