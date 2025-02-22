# model.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 从 config.py 中导入超参数
from config import (
    GAMMA, LAMBDA, CLIP_EPSILON, ENTROPY_BETA,
    LEARNING_RATE, BATCH_SIZE, EPOCHS
)
from config import (
    INITIAL_RADIUS, MIN_RADIUS, RADIUS_DECAY, SUCCESS_THRESHOLD, WINDOW_SIZE,
    EXPLORE_BONUS, DECAY_FACTOR
)

# ======================
# PPO模型
# ======================
class PPOActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(PPOActorCritic, self).__init__()
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
        logits = self.actor(x)
        probs = torch.softmax(logits, dim=-1)
        value = self.critic(x)
        return probs, value

# ======================
# 判别器模型（用于GAIL）
# ======================
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.net(sa)

# ======================
# 训练缓冲区
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
            torch.FloatTensor(self.states),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.dones)
        )

# ======================
# PPO训练器
# ======================
class PPOTrainer:
    def __init__(self, env, model, optimizer):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.buffer = PPOBuffer()
        self.success_history = []
        self.current_radius = INITIAL_RADIUS
        self.explore_bonus = EXPLORE_BONUS

    def update(self, success):
        self.env.current_radius = self.current_radius  # 确保环境同步
        self.env.explore_bonus = self.explore_bonus    # 同步探索奖励

        self.success_history.append(success)
        if len(self.success_history) > WINDOW_SIZE:
            self.success_history.pop(0)
        
        self.explore_bonus *= DECAY_FACTOR
        self.explore_bonus = max(self.explore_bonus, 0.1)  # 添加下限

        if len(self.success_history) >= WINDOW_SIZE:
            success_rate = np.mean(self.success_history[-WINDOW_SIZE:])
            if success_rate > SUCCESS_THRESHOLD:
                self.current_radius = max(
                    MIN_RADIUS,
                    self.current_radius * RADIUS_DECAY
                )
                print(f"Curriculum Update: radius -> {self.current_radius:.1f}")
                self.success_history = []
            elif success_rate < SUCCESS_THRESHOLD / 2:  # 如果成功率低于阈值的一半，扩大半径
                self.current_radius = min(
                    INITIAL_RADIUS,
                    self.current_radius / RADIUS_DECAY
                )
                print(f"Curriculum Update: radius -> {self.current_radius:.1f}")
                self.success_history = []

# ======================
# 判别器损失计算函数（用于GAIL）
# ======================
def compute_discriminator_loss(discriminator, expert_states, expert_actions, policy_states, policy_actions):
    # 将动作转换为 one-hot 编码
    action_dim = policy_actions.max().item() + 1  # 假设动作是从0开始的整数索引
    expert_actions_one_hot = torch.zeros(len(expert_actions), action_dim)
    expert_actions_one_hot[range(len(expert_actions)), expert_actions] = 1.0

    policy_actions_one_hot = torch.zeros(len(policy_actions), action_dim)
    policy_actions_one_hot[range(len(policy_actions)), policy_actions] = 1.0

    # 判别器预测
    expert_predictions = discriminator(expert_states, expert_actions_one_hot)
    policy_predictions = discriminator(policy_states, policy_actions_one_hot)

    # 判别器损失
    criterion = nn.BCELoss()
    expert_loss = criterion(expert_predictions, torch.ones_like(expert_predictions))
    policy_loss = criterion(policy_predictions, torch.zeros_like(policy_predictions))

    total_loss = expert_loss + policy_loss
    return total_loss

# ======================
# 获取专家数据的函数
# ======================
def get_expert_data():
    import numpy as np
    data = np.load('expert_data.npz')
    expert_states = torch.FloatTensor(data['states'])
    expert_actions = torch.LongTensor(data['actions'])
    return expert_states, expert_actions

