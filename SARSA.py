import numpy as np
import gym
import time
import matplotlib.pyplot as plt

# 创建悬崖环境
env = gym.make('CliffWalking-v0')
n_states = env.observation_space.n  # 48个状态（4x12网格）
n_actions = env.action_space.n  # 4个动作（上0，右1，下2，左3）

# 核心算法参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率
episodes = 5000  # 训练轮数

# 初始化Q表
Q_table = np.zeros((n_states, n_actions))  #48*4

# 训练进度跟踪
rewards_list = []
steps_list = []

# 1. SARSA训练过程
print("开始训练...")
for episode in range(episodes):
    state = env.reset()[0]  # 适配新版Gym API
    total_reward = 0
    step_count = 0

    # 选择初始动作（ε-贪婪策略）
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()  # 随机探索
    else:
        #对于当前状态state，从 Q 表中找出价值最高的动作，返回该动作的索引。
        #Q_table[state]：取出 Q 表中对应state行的所有动作价值（长度为 4 的数组）
        action = np.argmax(Q_table[state])  # 选择当前最优动作

    # 2. 单轮训练循环
    done = False
    while not done:
        # 适配新版Gym API，接收5个返回值
        result = env.step(action)#执行动作并获取环境反馈
        next_state, reward, done, truncated, info = result
        #turncated表示 episode 是否因时间限制或外部条件提前终止
        # 如果truncated也需要考虑，可以将done设为done或truncated
        done = done or truncated

        # 选择下一个动作（使用相同策略）
        if np.random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q_table[next_state])

        # 3. SARSA核心更新公式
        td_target = reward + gamma * Q_table[next_state, next_action]
        td_error = td_target - Q_table[state, action]
        Q_table[state, action] += alpha * td_error

        # 更新状态和动作
        state = next_state
        action = next_action

        total_reward += reward
        step_count += 1

    # 记录本轮表现
    rewards_list.append(total_reward)
    steps_list.append(step_count)

    # 每50轮显示进度
    if episode % 50 == 0:
        print(f"轮次: {episode}/{episodes}, 步数: {step_count}, 总奖励: {total_reward}")

print("训练完成！")


# 4. 评估训练结果
def test_policy():
    state = env.reset()[0]
    done = False
    path = []

    print("\n最终策略路径：")
    while not done:
        path.append(state)
        action = np.argmax(Q_table[state])  # 总是选择最优动作
        result = env.step(action)
        state, _, done, truncated, _ = result
        done = done or truncated

    # 可视化路径
    grid = np.zeros((4, 12))
    for s in path:
        x = s // 12
        y = s % 12
        grid[x, y] = 1  # 标记路径

    # 标记特殊位置
    grid[3, 0] = 2  # 起点
    for i in range(1, 11):
        grid[3, i] = 3  # 悬崖
    grid[3, 11] = 4  # 终点

    # 打印网格地图
    symbols = {0: '墙', 1: '路', 2: '起', 3: '崖', 4: '终'}
    print("悬崖环境地图 (路=路径, 崖=悬崖, 起起点, 终=终点):")
    for i in range(4):
        row = ""
        for j in range(12):
            row += symbols[grid[i, j]]
        print(row)


# 5. 显示训练指标
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, episodes + 1), rewards_list)
plt.title("Total Reward Variation per Episode")
plt.xlabel("Training Episodes")
plt.ylabel("Total Reward")

plt.subplot(1, 2, 2)
plt.plot(range(1, episodes + 1), steps_list, color='orange')
plt.title("Step Count Variation to Reach the Goal")
plt.xlabel("Training Episodes")
plt.ylabel("Steps")
plt.tight_layout()

plt.show()

# 显示最终策略
test_policy()

# 关闭环境
env.close()