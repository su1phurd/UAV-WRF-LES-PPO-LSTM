# train_ppo.py

import os
import pandas as pd
import torch
import numpy as np
from config import (
    LEARNING_RATE, GRID_SIZE, GAMMA, LAMBDA, CLIP_EPSILON, ENTROPY_BETA,
    BATCH_SIZE, EPOCHS, MAX_STEPS
)
from environment import MethaneEnv
from model import PPOActorCritic, PPOBuffer, PPOTrainer, NetCDFWriter

def _update_model(buffer, model, optimizer):
    states, actions, rewards, values, log_probs, dones = buffer.get()
    
    # 计算GAE
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    next_value = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards)-1:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t] * next_non_terminal
        else:
            next_non_terminal = 1.0 - dones[t+1]
            next_value = values[t+1] * next_non_terminal
        
        delta = rewards[t] + GAMMA * next_value - values[t]
        advantages[t] = delta + GAMMA * LAMBDA * next_non_terminal * last_advantage
        last_advantage = advantages[t]
    
    # 归一化（添加稳定性检查）
    advantages = (advantages - advantages.mean())
    adv_std = advantages.std()
    if adv_std < 1e-6 or torch.isnan(adv_std):
        adv_std = 1.0
    advantages = advantages / (adv_std + 1e-6)
    returns = advantages + values
        
    # 参数更新
    for _ in range(EPOCHS):
        indices = torch.randperm(len(states))
        for idx in indices.split(BATCH_SIZE):
            if len(idx) == 0: continue
            
            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = log_probs[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]
            batch_values = values[idx]
            
            new_probs, new_values = model(batch_states)
            
            # 检查probs合法性
            if torch.isnan(new_probs).any():
                print("Invalid probs detected!")
                print("Input states:", batch_states)
                print("Model output probs:", new_probs)
                raise RuntimeError("NaN in probs")
            
            new_dist = torch.distributions.Categorical(new_probs)
            new_log_probs = new_dist.log_prob(batch_actions)
            
            # 策略损失
            ratio = (new_log_probs - batch_old_log_probs).exp()
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 值函数损失
            value_pred_clipped = batch_values + (new_values.squeeze() - batch_values).clamp(-CLIP_EPSILON, CLIP_EPSILON)
            value_loss = 0.5 * torch.max(
                (new_values.squeeze() - batch_returns).pow(2),
                (value_pred_clipped - batch_returns).pow(2)
            ).mean()   
               
            # 熵正则项
            entropy = -torch.sum(new_probs * torch.log(new_probs + 1e-8), dim=1).mean()
            
            total_loss = policy_loss + value_loss - ENTROPY_BETA * entropy
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

class RadiusTracker:
    """跟踪最小两个半径的成功数据"""
    def __init__(self):
        self.radius_history = []
        self.success_data = {}  # {radius: [episode_data]}
    
    def update(self, current_radius, episode_data, is_success):
        """更新半径数据"""
        if is_success:
            if current_radius not in self.success_data:
                self.success_data[current_radius] = []
            self.success_data[current_radius].append(episode_data)
            
            # 维护历史记录仅保留最小两个半径
            if current_radius not in self.radius_history:
                self.radius_history.append(current_radius)
                self.radius_history.sort()
                if len(self.radius_history) > 2:
                    del self.radius_history[-1]

def train_ppo():
    """主训练函数"""
    env = MethaneEnv()
    model = PPOActorCritic(6, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = PPOBuffer()
    trainer = PPOTrainer(env, model, optimizer)
    radius_tracker = RadiusTracker()

    # 初始化NetCDF写入器
    nc_writer = NetCDFWriter(
        filename='training_data.nc',
        grid_size=GRID_SIZE,
        max_episodes=2000,
        max_steps=MAX_STEPS
    )

    # 训练数据收集
    training_data = []
    columns = [
        'Episode', 'Total_Reward', 'Success',
        'Conc_Reward', 'Explore_Reward',
        'Move_Penalty', 'TKE_Penalty', 
        'Boundary_Penalty', 'Steps', 'Final_Conc',
        'Current_Radius'
    ]

    # 训练主循环
    for episode in range(2000):
        state = env.reset()
        done = False
        episode_data = {
            'total_reward': 0,
            'steps': 0,
            'x': [], 'y': [], 'conc': [],
            'success': False,
            'source_conc': 0.0,
            'source_x':0.0,
            'source_y':0.0,
            'current_radius': trainer.current_radius,
            'Conc_Reward': 0.0,
            'Explore_Reward': 0.0,
            'Move_Penalty': 0.0,
            'TKE_Penalty': 0.0,
            'Boundary_Penalty': 0.0
        }

        while not done:
            # 策略选择动作
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs, value = model(state_t)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
            # 环境交互
            next_state, reward, done, info = env.step(action)
            # 记录轨迹数据和奖励分量
            x, y = env.agent_pos
            current_conc = env.conc_field[
                np.clip(int(x), 0, GRID_SIZE-1), 
                np.clip(int(y), 0, GRID_SIZE-1)
            ]
            episode_data['x'].append(float(x))
            episode_data['y'].append(float(y))
            episode_data['conc'].append(float(current_conc))
            episode_data['total_reward'] += reward
            episode_data['steps'] += 1
            episode_data['Conc_Reward'] += info['concentration_reward']
            episode_data['Explore_Reward'] += info['explore_reward']
            episode_data['Move_Penalty'] += info['move_penalty']
            episode_data['TKE_Penalty'] += info['tke_penalty']
            episode_data['Boundary_Penalty'] += info['boundary_penalty']
            # 存储经验到缓冲区
            buffer.store(
                state, action, reward,
                value.item(),
                action_dist.log_prob(torch.tensor(action)).item(),
                done
            )
            # 定期更新模型
            if len(buffer.states) >= BATCH_SIZE:
                _update_model(buffer, model, optimizer)
                buffer.clear()
            state = next_state
        # === 自主停止时记录源头浓度 ===
        if env.trajectory[-1]['reached']:
            final_pos = np.clip(env.agent_pos.astype(int), 0, GRID_SIZE-1)
            episode_data['source_conc'] = env.conc_field[final_pos[0], final_pos[1]]
            episode_data['source_x'] = float(env.agent_pos[0])
            episode_data['source_y'] = float(env.agent_pos[1])
            episode_data['success'] = True
        # === 更新半径跟踪器 ===
        radius_tracker.update(
            current_radius=trainer.current_radius,
            episode_data=episode_data,
            is_success=episode_data['success']
        )
        # === 只保存最小两个半径的成功数据到NetCDF ===
        if (trainer.current_radius in radius_tracker.radius_history and 
            episode_data['success']):
            nc_writer.write_episode_data(
                episode_idx=episode,
                steps=episode_data['steps'],
                x=np.array(episode_data['x']),
                y=np.array(episode_data['y']),
                conc=np.array(episode_data['conc']),
                source_x=episode_data['source_x'],
                source_y=episode_data['source_y'],
                source_conc=episode_data['source_conc'],
                sigma=env.gaussian_params['sigma'],
                peak=env.gaussian_params['peak']
            )
        # ...在每个episode结束后写入数据...
        nc_writer.write_episode_data(
            episode_idx=episode,
            steps=len(episode_data['x']),
            x=np.array(episode_data['x']),
            y=np.array(episode_data['y']),
            conc=np.array(episode_data['conc']),
            source_x=env.gaussian_params['mu_x'],
            source_y=env.gaussian_params['mu_y'],
            source_conc=env.gaussian_params['peak'],
            sigma=env.gaussian_params['sigma'],
            peak=env.gaussian_params['peak']
        )

        # === 保存训练统计数据 ===
        training_data.append([
            episode+1,
            episode_data['total_reward'],
            int(episode_data['success']),
            episode_data['Conc_Reward'],
            episode_data['Explore_Reward'],
            episode_data['Move_Penalty'],
            episode_data['TKE_Penalty'],
            episode_data['Boundary_Penalty'],
            episode_data['steps'],
            episode_data['source_conc'],
            trainer.current_radius
        ])

        # === 课程学习更新 ===
        trainer.update(episode_data['success'])

        # 打印进度
        if (episode+1) % 50 == 0:
            print(f"Ep {episode+1} | "
                  f"Reward: {episode_data['total_reward']:.1f} | "
                  f"Radius: {trainer.current_radius:.1f} | "
                  f"Success: {episode_data['success']}")

    # === 保存模型和训练结果 ===
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), 'model/ppo_successful_models.pth')
    df = pd.DataFrame(training_data, columns=columns)
    df.to_csv("training_results2_0.csv", index=False)
    nc_writer.close()
    print("\n训练完成！模型已保存为 ppo_successful_models.pth")
    print("训练数据已保存至 training_results2_0.csv 和 training_data.nc")

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    train_ppo()
