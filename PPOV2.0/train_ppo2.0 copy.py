# train_ppo_with_stopping.py

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pandas as pd
import torch
import numpy as np

# 导入超参数
from config import (
    LEARNING_RATE, GRID_SIZE, GAMMA, LAMBDA, CLIP_EPSILON, ENTROPY_BETA,
    BATCH_SIZE, EPOCHS
)

# 导入环境和模型
from environment import MethaneEnv
from model import PPOActorCritic, PPOBuffer, PPOTrainer

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

def train_ppo():
    """主训练函数"""
    env = MethaneEnv()
    model = PPOActorCritic(6, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = PPOBuffer()
    trainer = PPOTrainer(env, model, optimizer)

    #保存成功模型的列表
    successful_models = []
    
    # 数据收集结构
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
            'total': 0,
            'conc': 0,
            'explore': 0,
            'move_pen': 0,
            'tke_pen': 0,
            'boundary_pen': 0,
            'steps': 0,
            'final_conc': 0
        }

        while not done:
            # 策略选择动作
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs, value = model(state_t)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
            log_prob = action_dist.log_prob(torch.tensor(action))
            
            # 环境交互
            next_state, reward, done, info = env.step(action)
            
            # 存储经验到缓冲区
            buffer.store(
                state, 
                action, 
                reward, 
                value.item(),
                log_prob.item(),
                done
            )
            
            # 定期更新模型
            if len(buffer.states) >= BATCH_SIZE:
                _update_model(buffer, model, optimizer)
                buffer.clear()
            
            # 记录数据
            episode_data['total'] += reward
            episode_data['conc'] += info['concentration_reward']
            episode_data['explore'] += info['explore_reward']
            episode_data['move_pen'] += info['move_penalty']
            episode_data['tke_pen'] += info['tke_penalty']
            episode_data['boundary_pen'] += info['boundary_penalty']
            state = next_state

        # 处理剩余经验
        if len(buffer.states) > 0:
            _update_model(buffer, model, optimizer)
            buffer.clear()
        
        # 记录最终状态
        final_pos = np.clip(env.agent_pos.astype(int), 0, GRID_SIZE-1)
        episode_data['final_conc'] = env.conc_field[final_pos[0], final_pos[1]]
        episode_data['steps'] = env.step_count
        
        # 保存当前episode数据
        training_data.append([
            episode+1,
            episode_data['total'],
            int(env.trajectory[-1]['reached']),
            episode_data['conc'],
            episode_data['explore'],
            episode_data['move_pen'],
            episode_data['tke_pen'],
            episode_data['boundary_pen'],
            episode_data['steps'],
            episode_data['final_conc'],
            trainer.current_radius
        ])

        # 保存成功模型
        if env.trajectory[-1]['reached']:
            successful_models.append(model.state_dict())
        
        # 课程学习更新
        trainer.update(env.trajectory[-1]['reached'])
        
        # 打印进度
        if (episode+1) % 50 == 0:
            print(f"Ep {episode+1} | "
                  f"Reward: {episode_data['total']:.1f} | "
                  f"Conc: {episode_data['conc']:.1f} | "
                  f"Steps: {episode_data['steps']}")

    # 保存数据
    df = pd.DataFrame(training_data, columns=columns)
    df['Success_Rate'] = df['Success'].expanding().mean()
    df.to_csv("training_results1_3.csv", index=False)
    torch.save(model.state_dict(), 'model/ppo_successful_models.pth')
    print("\n训练完成！模型已保存为ppo_successful_models.pth，数据已保存至training_results1_4.csv")

if __name__ == '__main__':
    train_ppo()