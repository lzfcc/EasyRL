'''
In vscode Press Ctrl+Shift+P then type:
Python: Select Interpreter
'''

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import math
env = gym.make("CartPole-v1")  # , render_mode="human")

# Hyperparamters
EPISODES = 20000
DISCOUNT = 0.95
EPISODE_DISPLAY = 500
LEARNING_RATE = 0.25
EPSILON = 0.2

# env.observation
# cart position: [-4.8, 4.8], cart velocity:(-inf, inf), pole angle: [-0.42, 0.42], pole angular velocity: (-inf, inf)
# env.action_space: Discrete(2)
# Q-Table of size theta_state_size*theta_dot_state_size*env.action_space.n
theta_max = env.observation_space.high[2]  # 0.42
theta_min = env.observation_space.low[2]  # -0.42

theta_dot_minmax = math.radians(50)

# This corresponded to the number of divisions in the pole position and velocity space which was continuous.
theta_state_size = 50
theta_dot_state_size = 50

theta_window = (theta_max - theta_min) / theta_state_size
theta_dot_window = (theta_dot_minmax - (-theta_dot_minmax)
                    ) / theta_dot_state_size

Q_TABLE = np.random.randn(
    theta_state_size, theta_dot_state_size, env.action_space.n)

# For states
ep_rewards = []
ep_rewards_table = {'ep': [], 'avg': [], 'min': [], 'max': []}


def discretised_state(state):
    # state[2] -> theta
    # state[3] -> theta_dot
    discrete_state = np.array([0, 0])  # Initialised discrete array
    discrete_state[0] = (state[2] - theta_min) // theta_window
    discrete_state[0] = min(theta_state_size-1, max(0, discrete_state[0]))

    discrete_state[1] = (state[3] - (-theta_dot_minmax)) // theta_dot_window
    discrete_state[1] = min(theta_dot_state_size-1, max(0, discrete_state[1]))

    return tuple(discrete_state.astype(np.int32))


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

    if np.random.random() > EPSILON:
        action = np.argmax(Q_TABLE[curr_discrete_state])  # exploit
    else:
        action = np.random.randint(0, env.action_space.n)  # explore

    while not done and not truncated: # You must add not truncated otherwise the reward can be wrong
        new_state, reward, done, truncated, _info = env.step(action)
        new_discrete_state = discretised_state(new_state)

        if np.random.random() > EPSILON:
            new_action = np.argmax(Q_TABLE[new_discrete_state])
        else:
            new_action = np.random.randint(0, env.action_space.n)

        if render_state:
            env.render()

        '''sarsa'''
        # if not done:
        #     current_q = Q_TABLE[curr_discrete_state+(action,)] # <==> [curr_discrete_state[0],curr_discrete_state[1], action]
        #     max_future_q = Q_TABLE[new_discrete_state+(new_action,)]
        #     Q_TABLE[curr_discrete_state+(action,)] = current_q + \
        #         LEARNING_RATE*(reward+DISCOUNT*max_future_q-current_q)
        '''Q-Learning'''
        if not done:
            current_q = Q_TABLE[curr_discrete_state+(action,)]
            max_future_q = np.max(Q_TABLE[new_discrete_state])
            Q_TABLE[curr_discrete_state+(action,)] = current_q + \
                LEARNING_RATE*(reward + DISCOUNT*max_future_q - current_q)

        curr_discrete_state = new_discrete_state
        action = new_action

        episode_reward += reward

    ep_rewards.append(episode_reward)

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
plt.legend(loc=4)  # bottom right
plt.title('CartPole SARSA')
plt.ylabel('Average reward/Episode')
plt.xlabel('Episodes')
plt.show()
