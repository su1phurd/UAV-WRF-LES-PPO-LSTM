# evaluate_model.py

import numpy as np
import torch
import pandas as pd
from environment import MethaneEnv
from model import PPOActorCritic
from config import GRID_SIZE, CONC_PEAK

class ModelEvaluator:
    def __init__(self, model_path, eval_episodes=1000):
        self.env = MethaneEnv()
        self.model = self._load_model(model_path)
        self.eval_episodes = eval_episodes
        self.position_window = 10  # 位置稳定性检测窗口大小
        self.stability_threshold = 2.0  # 位置变化阈值（像素）
        self.conc_threshold = 0.8 * CONC_PEAK  # 浓度触发阈值

    def _load_model(self, path):
        model = PPOActorCritic(6, 5)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def _check_stop_condition(self, trajectory):
        """自主停止条件：连续10步位置变化小于阈值且浓度高于80%峰值"""
        if len(trajectory) < self.position_window:
            return False
        
        # 检测位置稳定性
        last_positions = [t['pos'] for t in trajectory[-self.position_window:]]
        pos_std = np.std(last_positions, axis=0).mean()
        
        # 检测浓度达标
        current_conc = trajectory[-1]['conc'] * CONC_PEAK
        
        return (pos_std < self.stability_threshold) and (current_conc > self.conc_threshold)

    def _calculate_deviation(self, agent_pos, source_pos):
        """计算偏差（欧氏距离）"""
        return np.linalg.norm(agent_pos - source_pos)

    def run_evaluation(self):
        results = []
        for ep in range(self.eval_episodes):
            state = self.env.reset()
            trajectory = []
            done = False
            steps = 0
            source_pos = self.env.source_pos  # 获取真实源位置
            
            while not done and steps < 2000:  # 最大步数保护
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    probs, _ = self.model(state_t)
                action = torch.argmax(probs).item()  # 贪婪策略
                
                next_state, _, done, info = self.env.step(action)
                trajectory.append({
                    'pos': self.env.agent_pos.copy(),
                    'conc': info['concentration_reward'] * CONC_PEAK
                })
                
                # 自主停止判断
                if self._check_stop_condition(trajectory):
                    done = True
                    print(f"Episode {ep+1}: 自主停止于步数 {steps}")

                state = next_state
                steps += 1

            # 计算偏差
            final_pos = trajectory[-1]['pos'] if trajectory else np.zeros(2)
            deviation = self._calculate_deviation(final_pos, source_pos)
            success = deviation < self.env.current_radius
            
            results.append({
                'episode': ep+1,
                'steps': steps,
                'deviation': deviation,
                'success': success,
                'final_conc': trajectory[-1]['conc'] if trajectory else 0
            })

        # 保存结果并生成报告
        df = pd.DataFrame(results)
        df.to_csv("evaluation_results.csv", index=False)
        success_rate = df['success'].mean()
        avg_deviation = df['deviation'].mean()
        print(f"验证完成！成功率: {success_rate:.2%}，平均偏差: {avg_deviation:.1f}像素")

if __name__ == '__main__':
    evaluator = ModelEvaluator("model/ppo_successful_models.pth", eval_episodes=1000)
    evaluator.run_evaluation()