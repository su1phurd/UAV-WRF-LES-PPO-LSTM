# train_ppo_gail.py

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

# 导入超参数
from config import (
    GAMMA, LAMBDA, CLIP_EPSILON, ENTROPY_BETA,
    LEARNING_RATE, BATCH_SIZE, EPOCHS
)

from config import (
    INITIAL_RADIUS, MIN_RADIUS, RADIUS_DECAY, SUCCESS_THRESHOLD, WINDOW_SIZE,
    EXPLORE_BONUS, DECAY_FACTOR
)

# 导入环境和模型
from environment import MethaneEnv
from model import (
    PPOActorCritic, Discriminator, PPOTrainer,
    PPOBuffer, compute_discriminator_loss, get_expert_data
)

# ======================
# 训练循环
# ======================
def train_ppo_gail():
    env = MethaneEnv()
    model = PPOActorCritic(input_size=6, output_size=5)
    discriminator = Discriminator(state_dim=6, action_dim=5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    buffer = PPOBuffer()
    trainer = PPOTrainer(env, model, optimizer)
    
    writer = SummaryWriter(log_dir='runs/ppo_gail_training')
    global_step = 0
    success_count = 0
    episode_rewards = []
    mean_rewards = []
    
    num_episodes = 2000  # 设置训练的总回合数

    for episode in range(num_episodes):
        # 将当前半径同步到环境
        env.current_radius = trainer.current_radius
        
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs, value = model(state_t)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
            log_prob = torch.log(probs[0, action])
            
            next_state, reward, done, _ = env.step(action)
            buffer.store(state, action, reward, value.item(), log_prob.item(), done)
            state = next_state
            total_reward += reward
            
            if len(buffer.states) >= BATCH_SIZE:
                states, actions, rewards, values, log_probs, dones = buffer.get()

                # 计算优势和回报
                with torch.no_grad():
                    next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
                    next_value = model(next_state_t)[1]
                next_value = next_value.squeeze(0)
                
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
                        old_log_p = log_probs[idx].detach()
                        adv = advantages[idx]
                        ret = returns[idx]
                        values_batch = values[idx].detach()
                        
                        # 将动作转换为one-hot向量
                        a_one_hot = torch.zeros(len(a), env.action_space.n)
                        a_one_hot[range(len(a)), a] = 1.0
                        
                        # 计算新的动作概率和价值估计
                        probs, new_values = model(s)
                        new_log_p = torch.log(probs.gather(1, a.unsqueeze(1)).squeeze())
                        
                        # 策略损失计算
                        ratio = (new_log_p - old_log_p).exp()
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * adv
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # 最大熵项
                        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                        policy_loss -= ENTROPY_BETA * entropy
                        
                        # 价值函数损失
                        value_clipped = values_batch + (new_values.squeeze() - values_batch).clamp(-CLIP_EPSILON, CLIP_EPSILON)
                        value_loss_original = (new_values.squeeze() - ret).pow(2)
                        value_loss_clipped = (value_clipped - ret).pow(2)
                        value_loss = 0.5 * torch.max(value_loss_original, value_loss_clipped).mean()
                        
                        # 总损失
                        total_loss = policy_loss + value_loss
                        
                        # 更新参数
                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()
                        
                        # 更新全局步骤计数器
                        global_step += 1
                
                buffer.clear()
        
        # 更新课程学习
        success = env.trajectory[-1]['reached']
        trainer.update(success)
        if success:
            success_count += 1
        
        # 使用专家数据训练判别器
        expert_states, expert_actions = get_expert_data()
        expert_actions_one_hot = torch.zeros(len(expert_actions), env.action_space.n)
        expert_actions_one_hot[range(len(expert_actions)), expert_actions] = 1.0
        
        policy_states = torch.FloatTensor(states)
        policy_actions_one_hot = torch.zeros(len(actions), env.action_space.n)
        policy_actions_one_hot[range(len(actions)), actions] = 1.0
        
        # 计算判别器的损失
        expert_pred = discriminator(expert_states, expert_actions_one_hot)
        policy_pred = discriminator(policy_states, policy_actions_one_hot)
        
        expert_loss = nn.BCELoss()(expert_pred, torch.ones_like(expert_pred))
        policy_loss_d = nn.BCELoss()(policy_pred, torch.zeros_like(policy_pred))
        discriminator_loss = expert_loss + policy_loss_d
        
        # 更新判别器
        optimizer_d.zero_grad()
        discriminator_loss.backward()
        optimizer_d.step()
        
        # 记录数据
        episode_rewards.append(total_reward)
        writer.add_scalar('Reward/Total', total_reward, episode)
        writer.add_scalar('Metrics/Success Rate', success_count / (episode + 1), episode)
        writer.add_scalar('Curriculum/Current Radius', trainer.current_radius, episode)
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param, episode)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, episode)
        
        # 打印训练信息
        if episode % 10 == 0:
            mean_reward = np.mean(episode_rewards[-10:])
            print(f'Episode {episode} | Mean Reward: {mean_reward:.2f} | Success Rate: {success_count / (episode + 1):.2%}')
    
    writer.close()

    # 保存模型
    torch.save(model.state_dict(), 'ppo_gail_model.pth')
    print('训练完成，模型已保存为 ppo_gail_model.pth')

if __name__ == '__main__':
    train_ppo_gail()

