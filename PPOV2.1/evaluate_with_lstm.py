#evaluate_with_lstm.py
import torch
import numpy as np
from environment import MethaneEnv
from model import PPOActorCritic
from sklearn.preprocessing import MinMaxScaler
import os
import torch.nn as nn
import check_gaussian

class PeakAndStopPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_peak = nn.Linear(hidden_dim, 1)
        self.fc_stop = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        peak = self.fc_peak(h).squeeze(-1)
        stop_prob = self.fc_stop(h).squeeze(-1)
        return peak, stop_prob

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ppo_model = PPOActorCritic(6, 5).to(device)
        ppo_model.load_state_dict(torch.load('model/ppo_successful_models.pth', map_location=device))
        ppo_model.eval()
        lstm_model = PeakAndStopPredictor(input_dim=1).to(device)
        lstm_model.load_state_dict(torch.load('model/best_peak_and_stop.pth', map_location=device))
        lstm_model.eval()
    except FileNotFoundError as e:
        print(f"模型加载失败: {e}")
        return

    window_size = 20
    env = MethaneEnv()
    metrics = {
        'deviations': [],
        'steps': [],
        'success': [],
        'stopped_early': [],
        'sigma_pred': [],
        'peak_pred': []
    }

    for ep in range(1000):
        state = env.reset()
        trajectory = []
        agent_xy_list = []
        done = False
        step_count = 0
        stopped_by_lstm = False
        stop_pos = None
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                probs, _ = ppo_model(state_t)
            action = torch.argmax(probs).item()
            next_state, _, done, _ = env.step(action)
            x, y = env.agent_pos
            current_conc = env.conc_field[int(x), int(y)]
            trajectory.append(current_conc)
            agent_xy_list.append([x, y])
            step_count += 1
            # LSTM判别是否到达源头附近
            if len(trajectory) >= window_size:
                input_seq = torch.FloatTensor(np.array(trajectory[-window_size:]).reshape(1, window_size, 1) / 100.0).to(device)
                with torch.no_grad():
                    peak_pred, stop_prob = lstm_model(input_seq)
                if stop_prob.item() > 0.8:
                    stopped_by_lstm = True
                    done = True
                    stop_pos = env.agent_pos.copy()
                    print(f"Episode {ep}: LSTM判定到达源头，step={step_count}，peak预测={peak_pred.item():.2f}")
            state = next_state
        final_pos = env.agent_pos
        source_pos = env.source_pos
        deviation = np.linalg.norm(final_pos - source_pos)
        metrics['deviations'].append(deviation)
        metrics['steps'].append(step_count)
        metrics['success'].append(deviation <= 50)
        metrics['stopped_early'].append(stopped_by_lstm)
        if stopped_by_lstm:
            metrics['sigma_pred'].append(np.nan)
            metrics['peak_pred'].append(peak_pred.item())
        else:
            metrics['sigma_pred'].append(np.nan)
            metrics['peak_pred'].append(np.nan)
        if (ep + 1) % 50 == 0:
            recent_success = np.mean(metrics['success'][-50:]) * 100
            avg_steps = np.mean(metrics['steps'][-50:])
            print(f"Ep {ep+1:03d} | 近期成功率 {recent_success:.1f}% | 平均步数 {avg_steps:.1f}")
            # 可视化当前episode，连线轨迹，白色方块标停止点
            check_gaussian.plot_gaussian_field(
                nc_file='training_data.nc',
                episode=ep % 1000,  # 防止越界
                stop_pos=stop_pos,
                traj_xy=np.array(agent_xy_list),
                save_path=f"results/gaussian_field_ep{ep+1}.png"
            )
    print("===== 综合验证结果 =====")
    print(f"平均定位偏差: {np.mean(metrics['deviations']):.2f} ± {np.std(metrics['deviations']):.2f} 像素")
    print(f"总体成功率: {np.mean(metrics['success'])*100:.1f}%")
    print(f"提前终止率: {np.mean(metrics['stopped_early'])*100:.1f}%")
    print(f"平均运行步数: {np.mean(metrics['steps']):.1f}")
    print(f"LSTM预测σ均值: {np.nanmean(metrics['sigma_pred']):.2f}，峰值均值: {np.nanmean(metrics['peak_pred']):.2f}")
    os.makedirs("results", exist_ok=True)
    np.savez("results/validation_metrics.npz", **metrics)

if __name__ == "__main__":
    main()