# environment.py to be updataed later

import numpy as np
import gym
from gym import spaces
from collections import defaultdict

# 从 config.py 中导入超参数
from config import (
    GRID_SIZE, MAX_STEPS, CONC_PEAK, TURBULENCE_INTENSITY,
    EXPLORE_BONUS, DECAY_FACTOR, GRID_DIVISIONS,
    INITIAL_RADIUS, MIN_RADIUS, RADIUS_DECAY, SUCCESS_THRESHOLD, WINDOW_SIZE
)

# ======================
# 环境类
# ======================
class MethaneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.action_space = spaces.Discrete(5)  # 定义动作空间：0-不动，1-上，2-下，3-右，4-左
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
        self.cell_size = self.grid_size // GRID_DIVISIONS
        self.visited = defaultdict(int)
        self.explore_bonus = EXPLORE_BONUS
        
        self.reset()

    def reset(self):
        # 生成新的源位置（避免靠近边界）
        padding = 50
        self.source_pos = np.random.rand(2) * (self.grid_size - 2 * padding) + padding
        self._generate_plume()
        
        # 固定初始位置为(0,0)
        self.agent_pos = np.array([0.0, 0.0])

        self.step_count = 0
        self.trajectory = []
        self.visited.clear()
        return self._get_obs()

    def _generate_plume(self):
        x, y = np.mgrid[:self.grid_size, :self.grid_size]
        dist = np.sqrt((x - self.source_pos[0])**2 + (y - self.source_pos[1])**2)
        base = CONC_PEAK * np.exp(-dist**2 / (2 * (self.grid_size / 16)**2))
        
        # 复杂湍流模式
        turbulence = TURBULENCE_INTENSITY * (
            np.abs(np.random.randn(self.grid_size, self.grid_size)) +
            0.3 * np.sin(0.05 * x) * np.cos(0.07 * y) +
            0.2 * np.random.rand(self.grid_size, self.grid_size)
        )
        self.conc_field = np.clip(base + turbulence, 0, CONC_PEAK)
        self.tke_field = turbulence  # 湍流动能场

    def _get_obs(self):
        x, y = self.agent_pos.astype(int)
        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)
        
        # 探索度计算
        grid_x = x // self.cell_size
        grid_y = y // self.cell_size
        visit_count = self.visited[(grid_x, grid_y)]
        explore_level = min(visit_count / 5.0, 1.0)  # 5次访问后视为已探索
        
        return np.array([
            self.agent_pos[0] / self.grid_size,       # 归一化的x位置
            self.agent_pos[1] / self.grid_size,       # 归一化的y位置
            self.conc_field[x, y] / CONC_PEAK,        # 归一化的浓度值
            self.tke_field[x, y] / (TURBULENCE_INTENSITY * 3),  # 归一化的湍流强度
            self.step_count / MAX_STEPS,              # 归一化的步数
            explore_level                             # 探索程度
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        
        # 动态移动步长（5%区域尺寸）
        move_step = self.grid_size * 0.05
        dx, dy = [
            (0, 0),                # 不动
            (0, move_step),        # 上
            (0, -move_step),       # 下
            (move_step, 0),        # 右
            (-move_step, 0)        # 左
        ][action]
        
        # 湍流影响
        x, y = self.agent_pos.astype(int)
        turbulence_effect = move_step * 0.2 * (
            np.random.randn(2) * self.tke_field[x, y] / (TURBULENCE_INTENSITY * 3)
        )
        new_pos = self.agent_pos + np.array([dx, dy]) + turbulence_effect
        
        # 弹性边界处理
        new_pos = np.clip(new_pos, -self.grid_size * 0.1, self.grid_size * 1.1)
        if np.any(new_pos < 0) or np.any(new_pos > self.grid_size):
            new_pos = self.agent_pos  # 撞击边界保持原位
        
        self.agent_pos = new_pos
        
        # 更新探索度
        grid_x = int(new_pos[0] // self.cell_size)
        grid_y = int(new_pos[1] // self.cell_size)

        self.visited[(grid_x, grid_y)] += 1  # 更新访问次数
        visit_count = self.visited[(grid_x, grid_y)]  # 获取更新后的访问次数
        explore_reward = self.explore_bonus / (visit_count + 1)
        
        # 获取观测
        obs = self._get_obs()
        
        # 计算基础奖励
        concentration_reward = 3.0 * obs[2]    # 浓度奖励
        turbulence_penalty = -0.2 * obs[3]     # 湍流惩罚
        movement_penalty = -0.1               # 移动惩罚
        base_reward = concentration_reward + turbulence_penalty + movement_penalty + explore_reward
        
        # 边界惩罚
        border_penalty = 0
        border_dist = min(
            new_pos[0], self.grid_size - new_pos[0],
            new_pos[1], self.grid_size - new_pos[1]
        )
        if border_dist < self.grid_size * 0.01:
            border_penalty = -0.5  # 增强边界惩罚
            base_reward += border_penalty
        
        # 到达奖励
        reach_reward = 0
        distance = np.linalg.norm(self.agent_pos - self.source_pos)
        reached = distance <= self.current_radius
        if reached:
            reach_reward = 100 * (INITIAL_RADIUS / self.current_radius)
            base_reward += reach_reward
        
        done = self.step_count >= MAX_STEPS or reached
        
        self.trajectory.append({
            'pos': self.agent_pos.copy(),
            'conc': obs[2],
            'tke': obs[3],
            'reached': reached
        })
        
        info = {
            'concentration_reward': concentration_reward,
            'turbulence_penalty': turbulence_penalty,
            'movement_penalty': movement_penalty,
            'explore_reward': explore_reward,
            'border_penalty': border_penalty,
            'reach_reward': reach_reward
        }
        
        return obs, base_reward, done, info

    def render(self, mode='human'):
        pass  # 如果需要，可实现环境的渲染方法

    def close(self):
        pass  # 如果需要，可在环境关闭时执行清理操作
