# evaluate_with_lstm.py
import torch
import numpy as np
from environment import MethaneEnv
from model import PPOActorCritic, ConcentrationThresholdPredictor
from sklearn.preprocessing import MinMaxScaler
import os
from config import (SUCCESS_DISTANCE_THRESHOLD, EVALUATE_SIZE)

class ThresholdController:
    def __init__(self, model, scaler, window_size=EVALUATE_SIZE):
        self.model = model
        self.scaler = scaler
        self.window_size = window_size
        self.current_threshold = None
        self.conc_buffer = []
        self.min_activate_steps = 2*EVALUATE_SIZE  # 最小激活步数

    def update_threshold(self, trajectory):
        if len(trajectory) >= max(self.window_size, self.min_activate_steps):
            valid_window = trajectory[-self.window_size:]
            scaled = self.scaler.transform(np.array(valid_window).reshape(-1, 1))
            with torch.no_grad():
                input_tensor = torch.FloatTensor(scaled).unsqueeze(0)  # 修正维度
                pred_source = self.model(input_tensor, lengths=[self.window_size])
            self.current_threshold = pred_source.item()* 0.95

    def should_stop(self, current_conc, step_count):
        self.conc_buffer.append(current_conc)
        if len(self.conc_buffer) > self.window_size:
            self.conc_buffer.pop(0)
        stop_condition = (
            step_count >= self.min_activate_steps and 
            self.current_threshold is not None and 
            (current_conc >= self.current_threshold or np.mean(self.conc_buffer) >= self.current_threshold)
        )
        return stop_condition

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ppo_model = PPOActorCritic(6, 5).to(device)
        ppo_model.load_state_dict(torch.load('model/ppo_successful_models.pth', map_location=device))
        ppo_model.eval()
        lstm_model = ConcentrationThresholdPredictor().to(device)
        lstm_model.load_state_dict(torch.load('model/lstm_threshold_predictor.pth', map_location=device))
        lstm_model.eval()
    except FileNotFoundError as e:
        print(f"模型加载失败: {str(e)}")
        return

    scaler = MinMaxScaler()
    try:
        scaler_params = np.load('model/scaler_params.npy')
        scaler.fit(scaler_params.reshape(-1, 1))
    except FileNotFoundError:
        print("标准化参数文件缺失")
        return

    env = MethaneEnv()
    controller = ThresholdController(lstm_model, scaler)

    metrics = {
        'deviations': [],
        'steps': [],
        'success': [],
        'stopped_early': []
    }

    for ep in range(1000):
        state = env.reset()
        trajectory = []
        done = False
        step_count = 0
        controller.conc_buffer = []
        stopped_early_flag = False  # 新增标志位

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                probs, _ = ppo_model(state_t)
            action = torch.argmax(probs).item()

            next_state, _, done, _ = env.step(action)
            x, y = env.agent_pos
            current_conc = env.conc_field[int(x), int(y)]
            trajectory.append(current_conc)
            step_count += 1

            if step_count % 10 == 0:
                controller.update_threshold(trajectory)

            if controller.should_stop(current_conc, step_count):
                stopped_early_flag = True
                done = True
                print(f"Episode {ep}: 自主停止于{step_count}步 | 当前浓度{current_conc:.1f}")
            state = next_state

        final_pos = env.agent_pos
        source_pos = env.source_pos
        deviation = np.linalg.norm(final_pos - source_pos)
        metrics['deviations'].append(deviation)
        metrics['steps'].append(step_count)
        metrics['success'].append(deviation <= SUCCESS_DISTANCE_THRESHOLD)
        metrics['stopped_early'].append(stopped_early_flag)  # 确保每个episode都有记录

        if (ep + 1) % 50 == 0:
            recent_success = np.mean(metrics['success'][-50:]) * 100
            avg_steps = np.mean(metrics['steps'][-50:])
            print(f"Ep {ep+1:03d} | 近期成功率 {recent_success:.1f}% | 平均步数 {avg_steps:.1f}")

    # 计算成功案例的平均偏差
    success_deviations = [
        deviation for deviation, success 
        in zip(metrics['deviations'], metrics['success']) 
        if success
    ]
    avg_success_deviation = np.mean(success_deviations) if success_deviations else 0
    std_success_deviation = np.std(success_deviations) if success_deviations else 0

    # 输出结果
    print("===== 综合验证结果 =====")
    print(f"平均定位偏差: {np.mean(metrics['deviations']):.2f} ± {np.std(metrics['deviations']):.2f} 像素")
    print(f"成功案例平均偏差: {avg_success_deviation:.2f} ± {std_success_deviation:.2f} 像素")  # 新增行
    print(f"总体成功率: {np.mean(metrics['success'])*100:.1f}%")
    # 修复提前终止率计算
    stopped_early_rate = np.mean(metrics['stopped_early']) * 100 if len(metrics['stopped_early']) > 0 else 0
    print(f"提前终止率: {stopped_early_rate:.1f}%")
    print(f"平均运行步数: {np.mean(metrics['steps']):.1f}")

    os.makedirs("results", exist_ok=True)
    np.savez("results/validation_metrics.npz", **metrics)

if __name__ == "__main__":
    main()