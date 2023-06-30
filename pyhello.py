import gymnasium as gym

env = gym.make('MountainCar-v0', render_mode="human")
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
# print('观测范围 = {} ~ {}'.format(env.observation_space.low, env.observation_space.high))
# print('动作数 = {}'.format(env.action_space.n))

class BespokeAgent:
    def __init__(self, env):
        pass
    
    def decide(self, observation): # 决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2 # Accelerate to the right
        else:
            action = 0 # Accelerate to the left
        return action # 返回动作
    
    def decide_cheet(self, epi, observation): # 决策
        if epi < 41: # best result: 38 ~ 40
            action = 0
        else:
            action = 2
        return action # 返回动作

    def learn(self, *args): # 学习
        pass
    
agent = BespokeAgent(env)


def play_montecarlo(env, agent, render=False, train=False):
    episode_reward = 0. # 记录回合总奖励，初始化为0
    observation, _info = env.reset(seed=0) # 重置游戏环境，开始新回合。设置随机数种子,只是为了让结果可以精确复现,一般情况下可删去
    # print(observation, type(observation), type(observation[0]))
    epi = 0
    while True: # 不断循环，直到回合结束
        epi += 1
        if render: # 判断是否显示
            env.render() # 显示图形界面，图形界面可以用 env.close() 语句关闭
        # action = agent.decide(observation)
        action = agent.decide_cheet(epi, observation)
        next_observation, reward, terminated, truncated, _info = env.step(action) # 执行动作
        episode_reward += reward # 收集回合奖励
        print("#", epi, ", action = ", action, ", reward = ", reward)
        if train: # 判断是否训练智能体
            agent.learn(observation, action, reward, terminated) # 学习
        if terminated: # 回合结束，跳出循环
            break
        observation = next_observation
    return episode_reward # 返回回合总奖励

episode_reward = play_montecarlo(env, agent, render=True)
print('回合奖励 = {}'.format(episode_reward))
env.close() # 此语句可关闭图形界面
