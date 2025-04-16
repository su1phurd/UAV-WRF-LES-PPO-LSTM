# train_lstm.py
import torch
import numpy as np
from config import TRAINING_SIZE
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from model import ConcentrationThresholdPredictor
from data_loader import load_raw_sequences
import os

class SequenceDataset(Dataset):
    def __init__(self, sequences, source_concs, TRAINING_SIZE):
        """
        修改：仅截取每个序列最后TRAINING_SIZE步作为训练样本
        """
        self.samples = []
        self.scaler = MinMaxScaler()
        all_windows = []
        
        # 收集所有末端窗口用于标准化
        for seq in sequences:
            if len(seq) >= TRAINING_SIZE:
                window = seq[-TRAINING_SIZE:]  # 只取最后TRAINING_SIZE步
                all_windows.append(window)
        
        # 全局标准化
        if all_windows:
            self.scaler.fit(np.concatenate(all_windows).reshape(-1, 1))
        
        # 生成样本
        for seq, source_conc in zip(sequences, source_concs):
            if len(seq) >= TRAINING_SIZE:
                window = seq[-TRAINING_SIZE:]  # 末端窗口
                window_scaled = self.scaler.transform(
                    np.array(window, dtype=np.float32).reshape(-1, 1)
                ).flatten()
                self.samples.append((
                    window_scaled,
                    source_conc  # 直接预测源头浓度
                ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.samples[idx][0]),  # [TRAINING_SIZE]
            torch.FloatTensor([self.samples[idx][1]])  # 源头浓度标签
        )

def train_lstm():
    # 加载数据
    sequences, source_concs = load_raw_sequences("training_data.nc")
    
    # 过滤过短序列
    valid_pairs = [
        (seq, conc) for seq, conc in zip(sequences, source_concs) 
        if len(seq) >= 10
    ]
    sequences, source_concs = zip(*valid_pairs) if valid_pairs else ([], [])
    
    dataset = SequenceDataset(sequences, source_concs, TRAINING_SIZE)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 初始化模型
    model = ConcentrationThresholdPredictor(input_size=1, hidden_size=128)
    criterion = nn.SmoothL1Loss(beta=2.0)  # 对异常值更敏感
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # 适当提高初始学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,  # 连续5轮无改善则降低学习率
    )    
    print("Current LR:", optimizer.param_groups[0]['lr'])


    # 训练循环
    for epoch in range(150):
        model.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(-1), lengths=[inputs.size(1)] * len(inputs))
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # 保存模型
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/lstm_threshold_predictor.pth")
    np.save("model/scaler_params.npy", dataset.scaler.data_min_)

if __name__ == "__main__":
    train_lstm()