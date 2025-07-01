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
episodes = 500  # 训练轮数

# 初始化Q表
Q_table = np.zeros((n_states, n_actions))  #48*4

# 训练进度跟踪
rewards_list = []
steps_list = []

# 1. Q学习训练过程
print("开始训练...")
for episode in range(episodes):
    state = env.reset()[0]  # 适配新版Gym API
    total_reward = 0
    step_count = 0

    # 单轮训练循环
    done = False
    while not done:
        # ε-贪婪策略选择动作
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 随机探索
        else:
            action = np.argmax(Q_table[state])  # 选择当前最优动作

        # 执行动作并获取环境反馈
        result = env.step(action)
        next_state, reward, done, truncated, info = result
        done = done or truncated

        # Q学习核心更新公式
        # 注意这里使用max Q(s',a')而不是实际执行的动作
        max_q_next = np.max(Q_table[next_state])#下一个状态价值最大动作的价值
        #我现在执行这个动作获得了即时奖励，再加上未来能获得的最大价值（打个折扣），这就是我执行这个动作的目标价值
        td_target = reward + gamma * max_q_next

        td_error = td_target - Q_table[state, action]
        Q_table[state, action] += alpha * td_error

        # 更新状态
        state = next_state

        total_reward += reward
        step_count += 1

    # 记录本轮表现
    rewards_list.append(total_reward)
    steps_list.append(step_count)

    # 每50轮显示进度
    # if episode % 50 == 0:
    #     print(f"轮次: {episode}/{episodes}, 步数: {step_count}, 总奖励: {total_reward}")

print("训练完成！")

# 2. 评估训练结果
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
    print("悬崖环境地图 (路=路径, 崖=悬崖, 起=起点, 终=终点):")
    for i in range(4):
        row = ""
        for j in range(12):
            row += symbols[grid[i, j]]
        print(row)

# 3. 显示训练指标
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, episodes + 1), rewards_list)
plt.title("每轮总奖励变化")
plt.xlabel("训练轮次")
plt.ylabel("总奖励")

plt.subplot(1, 2, 2)
plt.plot(range(1, episodes + 1), steps_list, color='orange')
plt.title("到达终点步数变化")
plt.xlabel("训练轮次")
plt.ylabel("步数")
plt.tight_layout()

plt.show()

# 显示最终策略
test_policy()

# 关闭环境
env.close()