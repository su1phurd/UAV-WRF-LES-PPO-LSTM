# model.py

import torch
import torch.nn as nn
import numpy as np
from netCDF4 import Dataset
from config import (
    INITIAL_RADIUS, MIN_RADIUS, RADIUS_DECAY, SUCCESS_THRESHOLD, WINDOW_SIZE,
    EXPLORE_BONUS, DECAY_FACTOR, TRAINING_SIZE, CONC_PEAK
)
import random

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
        for layer in self.feature:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        self.actor = nn.Linear(128, output_size)
        self.critic = nn.Linear(128, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)
    
    def forward(self, x):
        x = self.feature(x)
        logits = self.actor(x)
        if torch.isnan(logits).any():
            print("NaN in logits! Input:", x)
            raise RuntimeError("NaN in model output")
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
# 数据加载器
# ======================
def load_trajectory_segments(nc_path, tail_steps=60, window_size=20):
    with Dataset(nc_path, 'r') as nc:
        segments = []
        for ep in range(len(nc['episode'])):
            valid_steps = np.where(~np.isnan(nc['x'][ep]))[0]
            if len(valid_steps) < window_size:
                continue
            x_coords = nc['x'][ep, valid_steps]
            y_coords = nc['y'][ep, valid_steps]
            concentrations = nc['concentration'][ep, valid_steps]
            source_pos = np.array([nc['source_x'][ep], nc['source_y'][ep]])
            sigma = nc['gaussian_sigma'][ep] if 'gaussian_sigma' in nc.variables else 15.0
            # 滑动窗口采样，覆盖全轨迹
            for i in range(0, len(valid_steps) - window_size + 1):
                seg = {
                    'positions': np.column_stack((x_coords[i:i+window_size], y_coords[i:i+window_size])),
                    'concentrations': concentrations[i:i+window_size],
                    'source_pos': source_pos,
                    'sigma': sigma
                }
                segments.append(seg)
        print(f"Generated {len(segments)} segments (window_size={window_size})")
    return segments

def load_enhanced_samples(nc_path):
    with Dataset(nc_path) as nc:
        samples = []
        for ep in range(len(nc['episode'])):
            conc = nc['concentration'][ep][~np.isnan(nc['concentration'][ep])]
            x = nc['x'][ep][:len(conc)]
            y = nc['y'][ep][:len(conc)]
            sigma = nc['gaussian_sigma'][ep]
            peak = nc['peak_concentration'][ep]
            for i in range(WINDOW_SIZE, len(conc)):
                samples.append({
                    'window_conc': conc[i-WINDOW_SIZE:i],
                    'target': np.array([
                        nc['source_x'][ep],
                        nc['source_y'][ep],
                        sigma,
                        peak
                    ])
                })
        return samples

def calculate_dynamic_label(segment):
    conc = segment['concentrations']
    pos = segment['positions']
    src = segment['source_pos']
    
    dist = np.linalg.norm(pos[-1] - src)
    dist_score = np.exp(-dist/50.0)
    
    grad = np.gradient(conc)
    trend_score = np.tanh(np.mean(grad[-3:])/5.0)
    
    conc_score = np.clip(conc[-1]/CONC_PEAK, 0, 1)
    
    label = 0.4*dist_score + 0.3*(trend_score+1)/2 + 0.3*conc_score
    return np.clip(label, 0.01, 0.99)

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
        self.states.append(np.array(state, dtype=np.float32))  # 存储为numpy数组
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(float(done))

    def get(self):
        # 转换为numpy数组后再转张量
        states_np = np.stack(self.states, axis=0)
        actions_np = np.array(self.actions, dtype=np.int64)
        rewards_np = np.array(self.rewards, dtype=np.float32)
        values_np = np.array(self.values, dtype=np.float32)
        log_probs_np = np.array(self.log_probs, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.float32)

        return (
            torch.FloatTensor(states_np),
            torch.LongTensor(actions_np),
            torch.FloatTensor(rewards_np),
            torch.FloatTensor(values_np),
            torch.FloatTensor(log_probs_np),
            torch.FloatTensor(dones_np)
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
        self.env.current_radius = self.current_radius
        self.env.explore_bonus = self.explore_bonus

        self.success_history.append(success)
        if len(self.success_history) > WINDOW_SIZE:
            self.success_history.pop(0)
        
        # 动态探索衰减
        if len(self.success_history) >= WINDOW_SIZE:
            success_rate = np.mean(self.success_history[-WINDOW_SIZE:])
            self.explore_bonus *= (DECAY_FACTOR ** (1 + success_rate))
        
        self.explore_bonus = max(self.explore_bonus, 0.1)

        if len(self.success_history) >= WINDOW_SIZE:
            success_rate = np.mean(self.success_history[-WINDOW_SIZE:])
            if success_rate > SUCCESS_THRESHOLD:
                self.current_radius = max(
                    MIN_RADIUS,
                    self.current_radius * (RADIUS_DECAY ** (2 + 3*(success_rate-SUCCESS_THRESHOLD)))
                )
            elif success_rate < 0.25:
                self.current_radius = min(
                    INITIAL_RADIUS,
                    self.current_radius * 1.1
                )
            
            # 防震荡机制
            if abs(self.current_radius - self.env.current_radius) > 5:
                self.current_radius = self.env.current_radius + 5 * np.sign(self.current_radius - self.env.current_radius)

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

# ======================
# LSTM停止预测器
# ======================
class ConcentrationPredictor(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        return self.fc(out).squeeze(-1)

# ======================
# Gaussian LSTM
# ======================
class GaussianLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.mu_head = nn.Linear(hidden_size, 2)
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )
        self.peak_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        features = h_n[-1]
        return torch.cat([
            self.mu_head(features),
            self.sigma_head(features),
            self.peak_head(features)
        ], dim=1)

# ======================
# Gaussian Param Predictor
# ======================
class GaussianParamPredictor(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 输出mu_x, mu_y, sigma, peak
        )
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        return self.fc(out)

# ======================
# Gaussian Param And Stop Predictor
# ======================
class GaussianParamAndStopPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_sigma = nn.Linear(hidden_dim, 1)
        self.fc_peak = nn.Linear(hidden_dim, 1)
        self.fc_stop = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (batch, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        sigma = self.fc_sigma(h).squeeze(-1)
        peak = self.fc_peak(h).squeeze(-1)
        stop_prob = self.fc_stop(h).squeeze(-1)
        return sigma, peak, stop_prob

# ======================
# NetCDF 写入器
# ======================
class NetCDFWriter:
    def __init__(self, filename, grid_size, max_episodes=2000, max_steps=1000):
        self.filename = filename
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        
        self.ncfile = Dataset(filename, mode='w', format='NETCDF4')
        self.ncfile.createDimension('episode', max_episodes)
        self.ncfile.createDimension('step', max_steps)
        
        self.ncfile.GRID_SIZE = grid_size
        self._init_variables()
    
    def _init_variables(self):
        self.episode_var = self.ncfile.createVariable('episode', np.int32, ('episode',))
        self.episode_var.long_name = "Training episode index"
        
        self.step_var = self.ncfile.createVariable('step', np.int32, ('step',))
        self.step_var.long_name = "Step index within episode"
        
        self.x_var = self.ncfile.createVariable('x', np.float32, ('episode', 'step'), fill_value=np.nan, zlib=True)
        self.x_var.units = "grid unit"
        self.x_var.long_name = "Agent x-coordinate"
        
        self.y_var = self.ncfile.createVariable('y', np.float32, ('episode', 'step'), fill_value=np.nan, zlib=True)
        self.y_var.units = "grid unit"
        self.y_var.long_name = "Agent y-coordinate"
        
        self.conc_var = self.ncfile.createVariable('concentration', np.float32, ('episode', 'step'), fill_value=np.nan, zlib=True)
        self.conc_var.long_name = "Methane concentration"
        
        self.source_var = self.ncfile.createVariable('is_source', np.int8, ('episode', 'step'), fill_value=0, zlib=True)
        self.source_var.long_name = "Source position flag"
        
        self.source_conc_var = self.ncfile.createVariable('source_concentration', np.float32, ('episode',), fill_value=np.nan, zlib=True)
        self.source_conc_var.long_name = "Actual source concentration in each episode"
        
        self.source_x_var = self.ncfile.createVariable('source_x', np.float32, ('episode',), fill_value=np.nan, zlib=True)
        self.source_x_var.long_name = "Actual source x-coordinate"
        
        self.source_y_var = self.ncfile.createVariable('source_y', np.float32, ('episode',), fill_value=np.nan, zlib=True)
        self.source_y_var.long_name = "Actual source y-coordinate"

        self.sigma_var = self.ncfile.createVariable(
            'gaussian_sigma', np.float32, ('episode',)
        )
        self.sigma_var.long_name = "Gaussian distribution standard deviation"

        self.peak_var = self.ncfile.createVariable(
            'peak_concentration', np.float32, ('episode',)
        )
        self.peak_var.units = "ppm"
        self.peak_var.long_name = "Source peak concentration"
    
    def write_episode_data(self, episode_idx, steps, x, y, conc, source_x, source_y, source_conc, sigma, peak):
        self.x_var[episode_idx, :steps] = x
        self.y_var[episode_idx, :steps] = y
        self.conc_var[episode_idx, :steps] = conc
        
        self.source_var[episode_idx, steps-1] = 1
        self.x_var[episode_idx, steps-1] = source_x
        self.y_var[episode_idx, steps-1] = source_y
        
        self.source_conc_var[episode_idx] = source_conc
        self.source_x_var[episode_idx] = source_x
        self.source_y_var[episode_idx] = source_y

        self.sigma_var[episode_idx] = sigma
        self.peak_var[episode_idx] = peak
    
    def close(self):
        self.ncfile.close()