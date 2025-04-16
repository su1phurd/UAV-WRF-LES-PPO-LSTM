# generate_expert_data.py

import torch
import numpy as np

# 导入环境和模型
from environment import MethaneEnv
from model import PPOActorCritic

# 从 config.py 中导入必要的参数
from config import GRID_SIZE

# 初始化环境
env = MethaneEnv()

# 定义模型，确保输入和输出维度与训练时一致
model = PPOActorCritic(input_size=6, output_size=5)

# 加载训练好的模型参数
model.load_state_dict(torch.load('ppo_model.pth'))

# 将模型设置为评估模式
model.eval()

# 定义用于存储专家数据的列表
expert_states = []
expert_actions = []

# 运行一定数量的episodes，收集数据
num_episodes = 100

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            # 获取动作概率分布
            probs, _ = model(state_t)
        # 选择具有最高概率的动作
        action = torch.argmax(probs, dim=1).item()
        # 或者根据概率分布采样动作
        # action = torch.multinomial(probs, num_samples=1).item()
        
        # 存储状态和动作
        expert_states.append(state)
        expert_actions.append(action)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        state = next_state

print(f'已完成第 {episode + 1} 个episode的专家数据收集')

# 转换为NumPy数组
expert_states = np.array(expert_states)
expert_actions = np.array(expert_actions)

# 保存专家数据到文件
np.savez('expert_data.npz', states=expert_states, actions=expert_actions)
print('专家数据已保存到 expert_data.npz')
