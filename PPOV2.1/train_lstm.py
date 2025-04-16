import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import load_trajectory_segments
from config import GRID_SIZE

class TrajectoryDataset(Dataset):
    def __init__(self, segments, stop_radius=10, window_size=20):
        from config import GRID_SIZE
        self.grid_size = GRID_SIZE
        self.max_sigma = self._get_max_sigma(segments)
        self.max_peak = self._get_max_peak(segments)
        self.stop_radius = stop_radius
        self.window_size = window_size
        self.segments = segments
        self.features, self.labels = self._preprocess()
        
    def _get_max_sigma(self, segments):
        return max([seg.get('sigma', 15.0) for seg in segments]) if segments else 50.0
    
    def _get_max_peak(self, segments):
        return max([seg['concentrations'][-1] for seg in segments]) if segments else 100.0

    def _preprocess(self):
        import random
        features = []
        labels = []
        # 随机抽取1000条episode
        episode_dict = {}
        for seg in self.segments:
            ep_id = tuple(seg['source_pos']) # 用源头坐标唯一标识episode
            if ep_id not in episode_dict:
                episode_dict[ep_id] = []
            episode_dict[ep_id].append(seg)
        selected_eps = random.sample(list(episode_dict.values()), min(1000, len(episode_dict)))
        for ep_segs in selected_eps:
            seg = ep_segs[0]  # 取该episode的第一个segment
            conc = np.array(seg['concentrations'])
            # 负样本：前window_size步
            if len(conc) >= self.window_size:
                neg_feat = conc[:self.window_size].reshape(-1, 1) / 100.0
                peak = conc[self.window_size-1]
                labels.append([
                    peak / 100.0,
                    0.0
                ])
                features.append(neg_feat)
            # 正样本：最后window_size步
            if len(conc) >= self.window_size:
                pos_feat = conc[-self.window_size:].reshape(-1, 1) / 100.0
                last_pos = seg['positions'][-1]
                src = seg['source_pos']
                stop_label = 1.0 if np.linalg.norm(last_pos - src) <= self.stop_radius else 0.0
                peak = conc[-1]
                labels.append([
                    peak / 100.0,
                    stop_label
                ])
                features.append(pos_feat)
        print(f"采集到训练样本数: {len(labels)} (正样本+负样本)")
        return features, np.array(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor(self.labels[idx])
        )

def train():
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segments = load_trajectory_segments("training_data.nc", tail_steps=60)
    dataset = TrajectoryDataset(segments, window_size=20)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 模型与优化器
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

    model = PeakAndStopPredictor(input_dim=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    best_loss = float('inf')
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            peak_pred, stop_pred = model(features)
            loss_peak = nn.MSELoss()(peak_pred, labels[:,0])
            loss_stop = nn.BCELoss()(stop_pred, labels[:,1])
            loss = loss_peak + loss_stop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            torch.save(model.state_dict(), "model/best_peak_and_stop.pth")
            best_loss = avg_loss
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

if __name__ == "__main__":
    # 统计并可视化标签分布（以sigma为例，已归一化）
    import matplotlib.pyplot as plt
    segments = load_trajectory_segments("training_data.nc", tail_steps=60)
    dataset = TrajectoryDataset(segments, window_size=20)
    if len(dataset.labels) == 0:
        print("未采集到任何训练样本，请检查window_size设置或数据内容！")
        exit(1)
    plt.hist([l[0] for l in dataset.labels], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Sigma (normalized)')
    plt.ylabel('Count')
    plt.title('Distribution of Gaussian Sigma (normalized)')
    plt.savefig('sigma_distribution.png')
    print(f"sigma归一化统计：min={np.min([l[0] for l in dataset.labels]):.3f}, max={np.max([l[0] for l in dataset.labels]):.3f}, mean={np.mean([l[0] for l in dataset.labels]):.3f}, std={np.std([l[0] for l in dataset.labels]):.3f}")
    # 继续正常训练
    train()