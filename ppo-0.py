import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
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
CLIP_EPSILON = 0.2 #策略更新剪切范围
ENTROPY_BETA = 0.01 #熵正则化系数
LEARNING_RATE = 1e-4 #学习率
BATCH_SIZE = 256 #每次更新使用的样本量
EPOCHS = 5 #每次采样的训练轮次

# 探索参数
EXPLORE_BONUS = 1.0     # 基础探索奖励
DECAY_FACTOR = 0.995    # 探索衰减
GRID_DIVISIONS = 10     # 探索网格划分

# 课程学习参数
INITIAL_RADIUS = 50.0 #初始目标判定半径
MIN_RADIUS = 5.0 #初始目标判定半径
RADIUS_DECAY = 0.95 #半径衰减系数
SUCCESS_THRESHOLD = 0.7 #触发收缩的成功率阈值
WINDOW_SIZE = 50 #成功率计算窗口

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
        self.grid_size = GRID_SIZE
        self.action_space = spaces.Discrete(5)
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
        self.cell_size = GRID_SIZE // GRID_DIVISIONS
        self.visited = defaultdict(int)
        self.explore_bonus = EXPLORE_BONUS
        
        self.reset()

    def reset(self):
        # 生成新的源位置
        self.source_pos = np.random.rand(2) * (self.grid_size - 100) + 50
        self._generate_plume()
        
        # 固定初始位置为(0,0)
        self.agent_pos = np.array([0.0, 0.0])  # �޸Ĵ���
        
        self.step_count = 0
        self.trajectory = []
        self.visited.clear()
        return self._get_obs()

    def _generate_plume(self):
        x, y = np.mgrid[:self.grid_size, :self.grid_size]
        dist = np.sqrt((x - self.source_pos[0])**2 + (y - self.source_pos[1])**2)
        base = CONC_PEAK * np.exp(-dist**2/(2*(self.grid_size/16)**2))
        
        # 复杂湍流模式
        turbulence = TURBULENCE_INTENSITY * (
            np.random.randn(self.grid_size, self.grid_size) +
            0.3*np.sin(0.05*x) * np.cos(0.07*y) +
            0.2*np.random.rand(self.grid_size, self.grid_size)
        )
        self.conc_field = np.clip(base + turbulence, 0, CONC_PEAK)
        self.tke_field = np.abs(turbulence) * 2

    def _get_obs(self):
        x, y = self.agent_pos.astype(int)
        x = np.clip(x, 0, self.grid_size-1)
        y = np.clip(y, 0, self.grid_size-1)
        
        # 探索度计算
        grid_x = x // self.cell_size
        grid_y = y // self.cell_size
        visit_count = self.visited[(grid_x, grid_y)]
        explore_level = min(visit_count/5.0, 1.0)  # 5�η��ʺ���Ϊ��̽��
        
        return np.array([
            self.agent_pos[0]/self.grid_size,
            self.agent_pos[1]/self.grid_size,
            self.conc_field[x,y]/CONC_PEAK,
            self.tke_field[x,y]/(TURBULENCE_INTENSITY*3),
            self.step_count/MAX_STEPS,
            explore_level
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        
        # 动态移动步长（5%区域尺寸）
        move_step = self.grid_size * 0.05
        dx, dy = [(0,0), (0,move_step), (0,-move_step),
                 (move_step,0), (-move_step,0)][action]
        
        # 湍流影响
        x, y = self.agent_pos.astype(int)
        turbulence_effect = move_step * 0.2 * (
            np.random.randn(2) * self.tke_field[x,y]/(TURBULENCE_INTENSITY*3)
        )
        new_pos = self.agent_pos + [dx, dy] + turbulence_effect
        
        # 弹性边界处理
        new_pos = np.clip(new_pos, -self.grid_size*0.1, self.grid_size*1.1)
        if np.any(new_pos < 0) or np.any(new_pos > self.grid_size):
            new_pos = self.agent_pos  # ײ���߽籣��ԭλ
        
        self.agent_pos = new_pos
        
        # 探索奖励
        grid_x = int(new_pos[0] // self.cell_size)
        grid_y = int(new_pos[1] // self.cell_size)

        self.visited[(grid_x, grid_y)] += 1  # ���·��ʴ���
        visit_count = self.visited[(grid_x, grid_y)]  # ��ȡ���º�ķ��ʴ���
        explore_reward = self.explore_bonus / (visit_count + 1)
        
        # 基础奖励
        obs = self._get_obs()
        base_reward = (
            3.0 * obs[2]    # 浓度奖励
            - 0.3 * obs[3]  # 湍流惩罚
            - 0.05          # 移动惩罚
            + explore_reward
        )
        
        # 边界惩罚
        border_dist = min(
            new_pos[0], self.grid_size-new_pos[0],
            new_pos[1], self.grid_size-new_pos[1]
        )
        if border_dist < self.grid_size*0.1:
            base_reward -= 0.5  # 增强边界惩罚
        
        # 到达奖励
        distance = np.linalg.norm(self.agent_pos - self.source_pos)
        reached = distance <= self.current_radius
        if reached:
            base_reward += 100 * (INITIAL_RADIUS/self.current_radius)
        
        done = self.step_count >= MAX_STEPS or reached
        
        self.trajectory.append({
            'pos': self.agent_pos.copy(),
            'conc': obs[2],
            'tke': obs[3],
            'reached': reached
        })
        
        return obs, base_reward, done, {}

# ======================
# PPO模型
# ======================
class PPOActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, output_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.feature(x)
        probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return probs, value

# ======================
# 训练系统
# ======================
class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self):
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
        self.env = MethaneEnv()  # 修正为实例化环境
        self.success_history = []
        self.current_radius = INITIAL_RADIUS
        self.explore_bonus = EXPLORE_BONUS

    def update(self, success):
        self.env.current_radius = self.current_radius  # ȷ������ͬ��
        self.env.explore_bonus = self.explore_bonus    # ͬ��̽������

        self.success_history.append(success)
        if len(self.success_history) > WINDOW_SIZE:
            self.success_history.pop(0)
        
        self.explore_bonus *= DECAY_FACTOR
        self.explore_bonus = max(self.explore_bonus, 0.1)  # ��������

        if len(self.success_history) >= WINDOW_SIZE:
            success_rate = np.mean(self.success_history[-WINDOW_SIZE:])
            if success_rate > SUCCESS_THRESHOLD:
                self.current_radius = max(
                    MIN_RADIUS,
                    self.current_radius * RADIUS_DECAY
                )
                print(f"Curriculum Update: radius->{self.current_radius:.1f}")
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
    
    for episode in range(2000):
        # 将当前半径同步到环境
        env.current_radius = trainer.current_radius
        
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs, value = model(state_t)
            action = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[0, action])
            
            next_state, reward, done, _ = env.step(action)
            buffer.store(state, action, reward, value.item(), log_prob.item(), done)
            state = next_state
            total_reward += reward
            
            # 定期更新模型
            if len(buffer.states) >= BATCH_SIZE:
                states, actions, rewards, values, log_probs, dones = buffer.get()
                
                # 计算GAE
                # 添加一个终止状态的值估计
                with torch.no_grad():
                    next_value = model(torch.FloatTensor(next_state).unsqueeze(0))[1]
                next_value = next_value.squeeze(0)

                # 计算优势函数和回报（使用GAE）
                advantages = torch.zeros_like(rewards)
                returns = torch.zeros_like(rewards)
                gae = 0
                for t in reversed(range(len(rewards))):
                    if t == len(rewards) - 1:
                        next_non_terminal = 1.0 - dones[t]
                        next_value_t = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t+1]
                        next_value_t = values[t+1]
                    delta = rewards[t] + GAMMA * next_value_t * next_non_terminal - values[t]
                    gae = delta + GAMMA * LAMBDA * next_non_terminal * gae
                    advantages[t] = gae
                    returns[t] = advantages[t] + values[t]

                # 归一化优势函数
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                
                # 参数更新
                for _ in range(EPOCHS):
                    indices = torch.randperm(len(states))
                    for idx in indices.split(BATCH_SIZE):
                        if len(idx) == 0:
                            continue
                        
                        s = states[idx]
                        a = actions[idx]
                        old_log_p = log_probs[idx]
                        adv = advantages[idx]
                        ret = returns[idx]
                        
                        new_probs, new_values = model(s)
                        new_log_p = torch.log(new_probs.gather(1, a.unsqueeze(1))).squeeze()
                        ratio = (new_log_p - old_log_p).exp()
                        
                        # Clipped loss
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * adv
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss  CLIP_EPSILON?
                        value_pred_clipped = values + (new_values.squeeze() - values).clamp(-CLIP_EPSILON, CLIP_EPSILON)
                        value_losses = (new_values.squeeze() - ret).pow(2)
                        value_losses_clipped = (value_pred_clipped - ret).pow(2)
                        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                                                
                        # Entropy
                        entropy = -torch.sum(new_probs * torch.log(new_probs + 1e-8), dim=1).mean()
                        
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
        
        # 每10次计算平均奖励
        if episode % 10 == 9:
            mean_reward = np.mean(episode_rewards[-10:])
            mean_rewards.append((episode+1, mean_reward))  # ��¼��10n��λ��

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
            ax2.add_patch(plt.Circle(env.source_pos, env.current_radius,  # ��Ϊenv.current_radius
                                   color='yellow', fill=False, linestyle='--', 
                                   linewidth=2, alpha=0.8))
            ax2.scatter(env.source_pos[0], env.source_pos[1],
                      c='red', s=250, marker='*', edgecolor='gold', zorder=3)
            
            ax2.set_title(f'Episode {episode} (Radius: {env.current_radius:.1f})\n'  # ��ʾʵ��ʹ�ð뾶
                        f'Final Conc: {env.trajectory[-1]["conc"]*CONC_PEAK:.1f} ppm\n'
                        f'Steps: {len(env.trajectory)}/{MAX_STEPS}\n'
                        f'Success: {success_count}/{episode+1}')
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