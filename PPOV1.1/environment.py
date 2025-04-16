#environment.py

import numpy as np
import gym
from gym import spaces
from collections import defaultdict

# 从 config.py 中导入超参数
from config import (
    CONC_REWARD_COEF,
    GRID_SIZE, MAX_STEPS, CONC_PEAK, TURBULENCE_INTENSITY,
    EXPLORE_BONUS, GRID_DIVISIONS,
    INITIAL_RADIUS, MIN_RADIUS, RADIUS_DECAY, 
    TKE_PENALTY_FACTOR,BOUNDARY_PENALTY, BOUNDARY_DECAY_START,
)


class MethaneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.source_pos = None
        self.grid_size = GRID_SIZE
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # 初始化课程参数
        self.current_radius = INITIAL_RADIUS
        self.min_radius = MIN_RADIUS
        self.radius_decay = RADIUS_DECAY
        
        # 探索系统
        self.cell_size = self.grid_size // GRID_DIVISIONS
        self.visited = defaultdict(int)
        self.explore_bonus = EXPLORE_BONUS
        self.reset()

    def reset(self):
        padding = 50
        self.source_pos = np.random.rand(2) * (self.grid_size - 2 * padding) + padding
        self._generate_plume()
        self.agent_pos = np.array([0.0, 0.0])
        self.step_count = 0
        self.trajectory = []
        self.visited.clear()
        return self._get_obs()

    def _generate_plume(self):
        x, y = np.mgrid[:self.grid_size, :self.grid_size]
        dist = np.sqrt((x - self.source_pos[0])**2 + (y - self.source_pos[1])**2)
        base = CONC_PEAK * np.exp(-dist**2/(2*(self.grid_size/16)**2))
        
        turbulence = TURBULENCE_INTENSITY * (
            np.abs(np.random.randn(self.grid_size, self.grid_size)) +
            0.3*np.sin(0.05*x)*np.cos(0.07*y) +
            0.2*np.random.rand(self.grid_size, self.grid_size)
        )
        self.conc_field = np.clip(base + turbulence, 0, CONC_PEAK)
        self.tke_field = turbulence

    def _get_obs(self):
        x = np.clip(int(self.agent_pos[0]), 0, self.grid_size-1)
        y = np.clip(int(self.agent_pos[1]), 0, self.grid_size-1)
        
        grid_x = x // self.cell_size
        grid_y = y // self.cell_size
        visit_count = self.visited[(grid_x, grid_y)]
        explore_level = min(visit_count/5.0, 1.0)
        
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
        
        # 保存移动前的浓度
        prev_x = np.clip(int(self.agent_pos[0]), 0, self.grid_size-1)
        prev_y = np.clip(int(self.agent_pos[1]), 0, self.grid_size-1)
        prev_conc = self.conc_field[prev_x, prev_y] / CONC_PEAK
        
        # 移动逻辑
        move_step = self.grid_size * 0.05#调到0.01需要改奖励权重-折磨qaq
        dx, dy = [(0,0), (0,move_step), (0,-move_step),
                 (move_step,0), (-move_step,0)][action]
        move_magnitude = np.linalg.norm([dx, dy]) / (GRID_SIZE*0.05)
        move_penalty = -0.15 * (1 - move_magnitude)
        
        # 湍流影响计算
        x = np.clip(int(self.agent_pos[0]), 0, self.grid_size-1)
        y = np.clip(int(self.agent_pos[1]), 0, self.grid_size-1)
        turbulence_effect = move_step * 0.2 * (
            np.random.randn(2) * self.tke_field[x,y]/(TURBULENCE_INTENSITY*3))
        
        # 更新位置
        new_pos = self.agent_pos + [dx, dy] + turbulence_effect
        new_pos = np.clip(new_pos, 0, self.grid_size - 1e-6)
        self.agent_pos = new_pos.astype(np.float32)
        
        # === 边界惩罚计算 ===
        current_x = np.clip(int(new_pos[0]), 0, self.grid_size-1)
        current_y = np.clip(int(new_pos[1]), 0, self.grid_size-1)
        current_conc = self.conc_field[current_x, current_y] / CONC_PEAK
        conc_gradient = (current_conc - prev_conc) / (np.linalg.norm([dx, dy]) + 1e-6)
        
        boundary_dist = min(
            new_pos[0]/self.grid_size, 
            (self.grid_size - new_pos[0])/self.grid_size,
            new_pos[1]/self.grid_size,
            (self.grid_size - new_pos[1])/self.grid_size
        )
        
        if boundary_dist < BOUNDARY_DECAY_START and conc_gradient < -0.01:
            boundary_penalty = -BOUNDARY_PENALTY * (BOUNDARY_DECAY_START - boundary_dist)**2
        else:
            boundary_penalty = 0
        
        # 更新探索记录
        grid_x = int(new_pos[0] // self.cell_size)
        grid_y = int(new_pos[1] // self.cell_size)
        self.visited[(grid_x, grid_y)] += 1
        visit_count = self.visited[(grid_x, grid_y)]
        
        # 探索奖励计算
        explore_reward = (self.explore_bonus * (1 - self._get_obs()[5])) / (visit_count**0.75 + 1)
        
        # 获取观测值
        obs = self._get_obs()
        
        # 核心奖励计算
        total_reward = (
            CONC_REWARD_COEF * obs[2] +
            explore_reward +
            move_penalty -
            TKE_PENALTY_FACTOR * obs[3] +
            boundary_penalty
        )
        
        # 到达奖励
        distance = np.linalg.norm(self.agent_pos - self.source_pos)
        reached = distance <= self.current_radius
        if reached:
            total_reward += min(500,150 * (INITIAL_RADIUS / self.current_radius))
        
        done = self.step_count >= MAX_STEPS or reached
        self.trajectory.append({
            'pos': self.agent_pos.copy(),
            'conc': obs[2],
            'tke': obs[3],
            'reached': reached
        })
        
        info = {
            'concentration_reward': CONC_REWARD_COEF * obs[2],
            'explore_reward': explore_reward,
            'move_penalty': move_penalty,
            'tke_penalty': -TKE_PENALTY_FACTOR * obs[3],
            'boundary_penalty': boundary_penalty
        }
        
        return obs, total_reward, done, info