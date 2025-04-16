#ppo注释版
# ======================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import csv
from gym import spaces
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Rectangle

# ======================
# 超参数配置
# ======================
GRID_SIZE = 500        # 区域尺寸
MAX_STEPS = 1000       # 最大步数
CONC_PEAK = 100.0      # 峰值浓度
TURBULENCE_INTENSITY = 3.0  # 湍流强度

# PPO参数
GAMMA = 0.99  #未来奖励折扣因子
LAMBDA = 0.95 #GAE参数（平衡TD与MC）
CLIP_EPSILON = 0.2 #策略更新剪切范围，策略更新的变化幅度不能超过 20%
ENTROPY_BETA = 0.01 #熵正则化系数，化用于鼓励策略的探索性，通过增加策略的熵来防止过早收敛到次优解
LEARNING_RATE = 1e-4 #学习率
BATCH_SIZE = 256 #每次更新使用的样本量
EPOCHS = 5 #每次采样的训练轮次，每次采样后会进行 5 次参数更新

# 探索参数
EXPLORE_BONUS = 1.0     # 基础探索奖励，每次智能体探索新的状态时，会获得一个单位的奖励
DECAY_FACTOR = 0.995    # 探索衰减，每次更新时，探索奖励会乘以 0.995，从而逐渐减少探索奖励的影响
GRID_DIVISIONS = 10     # 探索网格划分，表示将环境划分为 10 x 10 的网格

# 课程学习参数
INITIAL_RADIUS = 50.0 #初始目标判定半径，设置智能体在训练初期的目标判定范围
MIN_RADIUS = 5.0 #最小目标判定半径，设置智能体在训练过程中目标判定范围的下限
RADIUS_DECAY = 0.95 #半径衰减系数，每次更新时，目标判定半径会乘以 0.95，从而逐渐减小
SUCCESS_THRESHOLD = 0.7 #触发收缩的成功率阈值，当智能体的成功率超过这个阈值时，目标判定半径会进行衰减
WINDOW_SIZE = 50 #成功率计算窗口，表示在计算成功率时，会考虑最近 50 次尝试的结果

#参数调优指南
#1. 训练速度慢
#提升探索：EXPLORE_BONUS=1.5, ENTROPY_BETA=0.05
#增大步长：move_step = GRID_SIZE*0.08
#简化课程：INITIAL_RADIUS=80, SUCCESS_THRESHOLD=0.6
#2. 策略震荡
#稳定更新：CLIP_EPSILON=0.1, LEARNING_RATE=5e-5
#增大批次：BATCH_SIZE=512
#增强裁剪：梯度裁剪阈值从0.5降至0.3
#3. 过早收敛
#增加探索：ENTROPY_BETA=0.03, DECAY_FACTOR=0.99
#复杂湍流：TURBULENCE_INTENSITY=4.0
#延迟收缩：WINDOW_SIZE=100, RADIUS_DECAY=0.97
#4. 边界徘徊
#增强惩罚：边界惩罚从-0.5增至-1.0
#添加激励：distance_reward = 0.2*(1 - distance/GRID_SIZE)
#限制移动：move_step = GRID_SIZE*0.03

# ======================
# 环境类
# ======================
class MethaneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5) #表示动作空间是一个离散的空间，包含 5 个可能的动作
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # 初始化课程学习参数
        self.current_radius = INITIAL_RADIUS
        self.min_radius = MIN_RADIUS
        self.radius_decay = RADIUS_DECAY
        self.success_threshold = SUCCESS_THRESHOLD
        
        # 探索系统
        self.grid_size = GRID_SIZE
        self.cell_size = GRID_SIZE // GRID_DIVISIONS #每个网格单元的大小
        self.visited = defaultdict(int) #用于记录智能体访问过的网格单元
        self.explore_bonus = EXPLORE_BONUS #探索奖励

        ##################
        self.conc_data = []  # 用于保存每步的浓度数据
        ##################
        
        self.reset()

    def reset(self): #用于重置环境到初始状态
        # 生成新的源位置
        self.source_pos = np.random.rand(2) * (self.grid_size - 100) + 50
        self._generate_plume()
        
        # 固定初始位置为(0,0)
        self.agent_pos = np.array([0.0, 0.0])  # 修改此行
        
        self.step_count = 0 #将步数计数器重置为 0，表示新的 episode 刚刚开始
        self.trajectory = [] #清空智能体的轨迹记录
        self.visited.clear() #清空访问记录

        #############
        # 每次调用 reset 时，清空浓度数据列表
        self.conc_data.clear()
        ################

        return self._get_obs()
    
    ############################
    def get_conc_data(self):
        """获取所有步的浓度数据"""
        return np.array(self.conc_data)

    def save_conc_data_to_csv(self,episode_data):
        """将每个episode的浓度数据保存到CSV文件"""
        with open('data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(episode_data)
    ############################

    def _generate_plume(self):
        x, y = np.mgrid[:self.grid_size, :self.grid_size] #生成一个网格坐标矩阵
        dist = np.sqrt((x - self.source_pos[0])**2 + (y - self.source_pos[1])**2) #计算每个网格点到源位置的距离
        base = CONC_PEAK * np.exp(-dist**2/(2*(self.grid_size/16)**2)) #算每个网格点的基础浓度值
        
        # 复杂湍流模式
        turbulence = TURBULENCE_INTENSITY * (
            np.random.randn(self.grid_size, self.grid_size) +
            0.3*np.sin(0.05*x) * np.cos(0.07*y) +
            0.2*np.random.rand(self.grid_size, self.grid_size)
        )#生成一个包含随机噪声和周期性波动的湍流矩阵
        self.conc_field = np.clip(base + turbulence, 0, CONC_PEAK) #计算最终的浓度场
        self.tke_field = np.abs(turbulence) * 2 # 计算湍流动能场

    def _get_obs(self):  #用于获取当前环境的观察状态
        x, y = self.agent_pos.astype(int)
        x = np.clip(x, 0, self.grid_size-1)
        y = np.clip(y, 0, self.grid_size-1)
        
        # 探索度计算
        grid_x = x // self.cell_size  #智能体所在的网格单元的坐标
        grid_y = y // self.cell_size
        visit_count = self.visited[(grid_x, grid_y)] #获取智能体访问该网格单元的次数
        explore_level = min(visit_count/5.0, 1.0)  # 5次访问后视为已探索
        

        ####################
        # 获取浓度数据（智能体当前位置的浓度值）
        conc_value = self.conc_field[x, y] / CONC_PEAK  # 当前浓度值（归一化）
        # 记录该步的浓度数据到列表中
        self.conc_data.append(conc_value)  # 将当前浓度值添加到浓度数据列表中
        ####################

        return np.array([
            self.agent_pos[0]/self.grid_size,  #智能体的位置，归一化到 [0, 1] 范围内
            self.agent_pos[1]/self.grid_size, 
            self.conc_field[x,y]/CONC_PEAK,    #智能体当前位置的浓度值
            self.tke_field[x,y]/(TURBULENCE_INTENSITY*3),  #智能体当前位置的湍流动能值
            self.step_count/MAX_STEPS,   #当前步数
            explore_level    #在当前网格单元的探索度
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1  #每执行一次step函数，就将步数计数器加 1
        
        # 动态移动步长（5%区域尺寸）
        move_step = self.grid_size * 0.05
        dx, dy = [(0,0), (0,move_step), (0,-move_step),
                 (move_step,0), (-move_step,0)][action] #根据动作选择对应的移动方向和步长
        
        # 湍流影响
        x, y = self.agent_pos.astype(int)
        turbulence_effect = move_step * 0.2 * (
            np.random.randn(2) * self.tke_field[x,y]/(TURBULENCE_INTENSITY*3)
        ) #计算湍流影响，湍流影响是一个随机噪声，乘以湍流动能场的值。
        new_pos = self.agent_pos + [dx, dy] + turbulence_effect
        #计算智能体的新位置

        # 弹性边界处理
        new_pos = np.clip(new_pos, -self.grid_size*0.1, self.grid_size*1.1)  #将新位置限制在合理范围内
        if np.any(new_pos < 0) or np.any(new_pos > self.grid_size):  #如果新位置超出边界
            new_pos = self.agent_pos  # 撞击边界保持原位
        
        self.agent_pos = new_pos
        
        # 探索奖励
        grid_x = int(new_pos[0] // self.cell_size) #计算智能体所在的网格单元的坐标
        grid_y = int(new_pos[1] // self.cell_size)

        self.visited[(grid_x, grid_y)] += 1  # 更新访问次数
        visit_count = self.visited[(grid_x, grid_y)]  # 获取更新后的访问次数
        explore_reward = self.explore_bonus / (visit_count + 1) #计算探索奖励，访问次数越多，探索奖励越少
        
        # 基础奖励
        obs = self._get_obs()  #获取当前观察状态
        base_reward = (   #计算基础奖励，包括浓度奖励、湍流惩罚、移动惩罚和探索奖励
            3.0 * obs[2]    # 浓度奖励
            - 0.3 * obs[3]  # 湍流惩罚
            - 0.05          # 移动惩罚
            + explore_reward
        )
        
        # 边界惩罚
        border_dist = min(
            new_pos[0], self.grid_size-new_pos[0],
            new_pos[1], self.grid_size-new_pos[1]
        )  #计算智能体到边界的最小距离
        if border_dist < self.grid_size*0.1:  #如果距离小于网格大小的 10%
            base_reward -= 0.5  # 增强边界惩罚
        
        # 到达奖励
        distance = np.linalg.norm(self.agent_pos - self.source_pos)  #计算智能体到源位置的距离
        reached = distance <= self.current_radius  #如果距离小于当前半径，表示到达目标
        if reached:
            base_reward += 100 * (INITIAL_RADIUS/self.current_radius) #增加到达奖励
        
        done = self.step_count >= MAX_STEPS or reached  #结束条件是达到最大步数或到达目标
        
        self.trajectory.append({
            'pos': self.agent_pos.copy(),
            'conc': obs[2],
            'tke': obs[3],
            'reached': reached
        })
        
        return obs, base_reward, done, {}  #返回观察状态、奖励、是否结束

# ======================
# PPO模型
# ======================
class PPOActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.feature = nn.Sequential( #定义了一个名为 feature 的特征提取网络，由一个 nn.Sequential 容器构成
            nn.Linear(input_size, 256), #线性层，将输入大小转换为 256
            nn.LayerNorm(256),  #层归一化层，用于标准化每个样本的特征
            nn.ReLU(),  #ReLU 激活函数，增加非线性
            nn.Linear(256, 128),  #另一个线性层，将 256 转换为 128
            nn.LayerNorm(128),  #另一个层归一化层
            nn.ReLU()  #另一个 ReLU 激活函数
        )
        self.actor = nn.Linear(128, output_size) #将128维的特征转换为输出大小output_size，用于生成策略分布
        self.critic = nn.Linear(128, 1)  #将128维的特征转换为一个标量值，用于估计状态价值

    def forward(self, x):
        x = self.feature(x)  #将原始输入转换为更高层次的特征表示
        probs = torch.softmax(self.actor(x), dim=-1)  #将输入张量转换为概率分布，确保输出的各个元素之和为 1
        value = self.critic(x)
        return probs, value #返回策略分布和状态价值

# ======================
# 训练系统
# ======================
class PPOBuffer:
    def __init__(self):  #初始化了这些列表
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []  #记录策略分布的对数概率
        self.dones = []  #记录每个样本的结束状态

    def clear(self): #清空这些列表
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def store(self, state, action, reward, value, log_prob, done):
        #将一个新的数据条目添加到各个列表中，每次执行一个动作后，都会调用这个方法
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self): #将所有存储的数据转换为 PyTorch 张量，并返回这些张量
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.dones)
        )

class PPOTrainer:
    def __init__(self):
        self.success_history = []  #用于记录每次训练的成功情况
        self.current_radius = INITIAL_RADIUS  #当前目标判定半径
        self.explore_bonus = EXPLORE_BONUS  #当前探索奖励

    def update(self, success):  #更新训练过程中的参数
        self.success_history.append(success)
        if len(self.success_history) > WINDOW_SIZE:  #如果成功历史记录的长度超过了窗口大小
            self.success_history.pop(0)  #删除最早的记录，保持窗口大小不变
        
        # 探索奖励衰减
        self.explore_bonus *= DECAY_FACTOR
        
        # 动态调整半径
        if len(self.success_history) >= WINDOW_SIZE:  #如果成功历史记录的长度大于等于窗口大小
            success_rate = np.mean(self.success_history[-WINDOW_SIZE:])  #计算最近 WINDOW_SIZE 次尝试的成功率
            if success_rate > SUCCESS_THRESHOLD:  #如果成功率超过了成功率阈值
                self.current_radius = max(  #动态调整，将当前半径设置为当前半径和最小半径的较大值
                    MIN_RADIUS,
                    self.current_radius * RADIUS_DECAY
                )
                print(f"New radius: {self.current_radius:.1f}")
                self.success_history = []

# ======================
# 训练循环
# ======================
def train_ppo():
    env = MethaneEnv()
    model = PPOActorCritic(6, 5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = PPOBuffer()
    trainer = PPOTrainer()
    
    # 初始化可视化
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    episode_rewards = []
    mean_rewards = []  # 存储每10次的平均奖励
    success_count = 0
    data = []

    # 动态经验筛选参数
    HIGH_REWARD_THRESHOLD = 80.0    # 高奖励阈值
    LOW_REWARD_THRESHOLD = 20.0     # 低奖励阈值
    REWARD_FILTER_START_EP = 300    # 开始筛选的episode
    PRIORITIZED_SAMPLING = True     # 是否启用优先级采样
    
    for episode in range(200):
        # 将当前半径同步到环境
        env.current_radius = trainer.current_radius
        
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs, value = model(state_t)
            action = torch.multinomial(probs, 1).item()   #从策略分布中采样一个动作 action
            log_prob = torch.log(probs[0, action])   #计算采样动作的对数概率
            
            next_state, reward, done, _ = env.step(action)  #执行动作 action，并从环境 env 中获取下一个状态 next_state
            buffer.store(state, action, reward, value.item(), log_prob.item(), done)  #将当前状态、动作、奖励、状态价值、对数概率和结束状态存储到经验缓冲区 buffer 中
            state = next_state  #更新当前状态为下一个状态
            total_reward += reward  #累加奖励
            
            # 定期更新模型
            if len(buffer.states) >= BATCH_SIZE:  #当缓冲区中的状态数量达到批量大小 BATCH_SIZE 时，进行模型参数更新
                states, actions, rewards, values, log_probs, dones = buffer.get()  #从缓冲区中获取存储的状态、动作、奖励、价值、对数概率和结束标志，并转换为 PyTorch 张量
                
                # 计算GAE，用于衡量当前策略相对于基准策略的优势
                advantages = torch.zeros_like(rewards)
                last_advantage = 0
                for t in reversed(range(len(rewards))):
                    if t < len(rewards) - 1:
                        next_value = values[t+1] * (1 - dones[t])
                    else:
                        next_value = 0
                    delta = rewards[t] + GAMMA * next_value - values[t]
                    advantages[t] = delta + GAMMA * LAMBDA * last_advantage * (1 - dones[t])
                    last_advantage = advantages[t]
                
                # 归一化
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                returns = advantages + values
                
                # 参数更新
                for _ in range(EPOCHS):
                    indices = torch.randperm(len(states))
                    for idx in indices.split(BATCH_SIZE):
                        if len(idx) == 0:
                            continue
                        
                        s = states[idx]  #从状态、动作、奖励、价值、对数概率和优势中获取当前批次的数据
                        a = actions[idx]
                        old_log_p = log_probs[idx]
                        adv = advantages[idx]
                        ret = returns[idx]
                        
                        new_probs, new_values = model(s)  #根据状态 s 获取新的策略分布和状态价值
                        new_log_p = torch.log(new_probs.gather(1, a.unsqueeze(1))).squeeze()  #计算新的对数概率
                        ratio = (new_log_p - old_log_p).exp()  #计算新旧对数概率的比值
                        
                        # Clipped loss
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * adv
                        policy_loss = -torch.min(surr1, surr2).mean()  #使用裁剪损失函数计算策略损失 policy_loss
                        
                        # Value loss
                        value_loss = 0.5 * (new_values.squeeze() - ret).pow(2).mean()
                        
                        # Entropy，计算熵损失 entropy，用于鼓励策略的探索性
                        entropy = -torch.sum(new_probs * torch.log(new_probs + 1e-8), dim=1).mean()
                        
                        #总损失 total_loss 为策略损失、价值损失和熵损失的加权和
                        total_loss = policy_loss + value_loss - ENTROPY_BETA * entropy
                        
                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()
                
                buffer.clear()
        
        # 更新课程学习
        success = env.trajectory[-1]['reached']
        trainer.update(success)
        if success:
            success_count += 1
        
        # 记录数据
        episode_rewards.append(total_reward)

        ###################
        conc_data = env.get_conc_data()  # 获取当前 episode 所有步的浓度数据
        data.append(conc_data)  # 将当前episode的数据添加到总数据中
        env.save_conc_data_to_csv(conc_data)
        ###################
        
        # 每10次计算平均奖励
        if episode % 10 == 9:
            mean_reward = np.mean(episode_rewards[-10:])
            mean_rewards.append((episode+1, mean_reward))  # 记录在10n的位置

        # 可视化更新（每10次）
        if episode % 10 == 0:
            ax1.clear()
            ax2.clear()
            
            # 左图：奖励分布
            if episode_rewards:
                # 绘制所有奖励散点
                ax1.scatter(range(len(episode_rewards)), episode_rewards, 
                          c='blue', alpha=0.4, s=20, label='Single Episode')
                
                # 绘制平均奖励连线
                if mean_rewards:
                    x_vals, y_vals = zip(*mean_rewards)
                    ax1.plot(x_vals, y_vals, 'r-', marker='o', markersize=8, 
                           linewidth=2, label='10-Episode Average')
                
                ax1.set_title(f'Training Progress (Radius: {trainer.current_radius:.1f})')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Reward')
                ax1.grid(True)
                ax1.legend(loc='upper left')

            # 右图：浓度场和路径
            conc = env.conc_field.T
            img = ax2.imshow(conc, origin='lower', cmap='viridis',
                           extent=[0, GRID_SIZE, 0, GRID_SIZE], alpha=0.8)
            
            # 绘制无人机轨迹（使用当前半径判断）
            traj = np.array([p['pos'] for p in env.trajectory])
            ax2.plot(traj[:,0], traj[:,1], 'w-', linewidth=1.5, alpha=0.8)
            # 终点标记使用实际判定结果
            final_reached = env.trajectory[-1]['reached']
            ax2.scatter(traj[-1,0], traj[-1,1], 
                      c='cyan' if final_reached else 'white',
                      s=120, edgecolors='black', zorder=3)
            
            # 使用同步后的当前半径绘制
            ax2.add_patch(plt.Circle(env.source_pos, env.current_radius,  # 改为env.current_radius
                                   color='yellow', fill=False, linestyle='--', 
                                   linewidth=2, alpha=0.8))
            ax2.scatter(env.source_pos[0], env.source_pos[1],
                      c='red', s=250, marker='*', edgecolor='gold', zorder=3)
            
            ax2.set_title(f'Episode {episode} (Radius: {env.current_radius:.1f})\n'  # 显示实际使用半径
                        f'Final Conc: {env.trajectory[-1]["conc"]*CONC_PEAK:.1f} ppm\n'
                        f'Steps: {len(env.trajectory)}/{MAX_STEPS}')
            ax2.set_xlim(0, GRID_SIZE)
            ax2.set_ylim(0, GRID_SIZE)
            
            plt.tight_layout()
            plt.pause(0.01)
                
        print(f'Ep {episode} | Radius: {env.current_radius:.1f} | Reward: {total_reward:.1f} | ' 
              f'10-Avg: {mean_rewards[-1][1] if mean_rewards else 0:.1f} | '
              f'Success: {success_count}/{episode+1}')

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    train_ppo()