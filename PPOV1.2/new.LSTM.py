#new.LSTM.py
#通过2000次的轮回训练产生的数据，学习使用前半部分的浓度数据去预测最终源头的浓度，得到一个大概的浓度范围，
#如果接下来遇到了一个大于该浓度范围的值，认为其为源头，
#如果过一段时间，仍未遇到源头，把使用后来又收集到的数据，再次预测源头的浓度，并且降低标准
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# 加载数据并预处理
def load_and_preprocess_data(data, new_data):
    # data为训练数据，new_data为用来预测的输入数据
    # 提取浓度数据和源头浓度
    concentration_data = data.iloc[:, :-1].values  # 所有列除了最后一列为浓度数据
    source_concentration = data.iloc[:, -1].values  # 最后一列为源头浓度

    # 标准化浓度数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    concentration_data = scaler.fit_transform(concentration_data)

    # 将数据拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(concentration_data, source_concentration, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler

# 转换为Tensor，并填充序列长度不一的数据
def create_data_loader(X, y, batch_size=32):
    X_tensor = [torch.tensor(x, dtype=torch.float32) for x in X]
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # 计算每个序列的真实长度
    sequence_lengths = [len(x) for x in X_tensor]
    
    # 使用pad_sequence对不同长度的数据进行填充
    X_padded = pad_sequence(X_tensor, batch_first=True, padding_value=0)
    
    dataset = TensorDataset(X_padded, y_tensor, torch.tensor(sequence_lengths, dtype=torch.long))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, lengths):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 使用pack_padded_sequence来忽略填充部分
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM层
        packed_out, _ = self.lstm(packed_input, (h0, c0))
        
        # 反向解包
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        # 取LSTM的最后一个时间步输出
        out = out[torch.arange(out.size(0)), lengths - 1, :]
        
        # 全连接层
        out = self.fc(out)
        
        return out

# 训练模型
def train_model(X_train, y_train, input_size, hidden_size, output_size, num_epochs=50, batch_size=32, learning_rate=0.001):
    train_loader = create_data_loader(X_train, y_train, batch_size)
    
    # 初始化模型、损失函数和优化器
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels, lengths in train_loader:
            # 将输入和标签移到GPU（如果有）
            inputs, labels = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            # 前向传播
            outputs = model(inputs, lengths)
            
            # 计算损失
            loss = criterion(outputs.squeeze(), labels)
            
            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 每个epoch输出一次损失
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader)}")
    
    return model

# 测试模型
def test_model(model, X_test, y_test):
    test_loader = create_data_loader(X_test, y_test, batch_size=32)
    model.eval()
    
    predicted_concentrations = []
    actual_concentrations = []
    
    with torch.no_grad():
        for inputs, labels, lengths in test_loader:
            inputs, labels = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(inputs, lengths)
            predicted_concentrations.extend(outputs.squeeze().cpu().numpy())
            actual_concentrations.extend(labels.cpu().numpy())
    
    return np.array(predicted_concentrations), np.array(actual_concentrations)

# 可视化结果
def visualize_results(predicted_concentrations, actual_concentrations):
    plt.figure(figsize=(10,6))
    plt.plot(actual_concentrations, label='Actual Concentration')
    plt.plot(predicted_concentrations, label='Predicted Concentration')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Concentration')
    plt.title('Predicted vs Actual Source Concentration')
    plt.show()

# 训练和测试流程的封装
def main(data_file, new_data_file, num_epochs=50):
    # 加载数据
    data = pd.read_csv(data_file)  # 用于训练的2000次轮回数据
    new_data = pd.read_csv(new_data_file)  # 用于预测的浓度数据
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data, new_data)
    
    # 模型超参数
    input_size = X_train.shape[1]  # 输入的特征数（即浓度数据的列数）
    hidden_size = 64
    output_size = 1  # 预测源头浓度为一个数值
    
    # 训练模型
    model = train_model(X_train, y_train, input_size, hidden_size, output_size, num_epochs)
    
    # 测试模型
    predicted_concentrations, actual_concentrations = test_model(model, X_test, y_test)
    
    # 可视化结果
    visualize_results(predicted_concentrations, actual_concentrations)

# 如果你要在其他地方调用，只需要运行main函数
if __name__ == "__main__":
    data_file = 'data.csv'
    new_data_file = 'new-data.csv'
    main(data_file, new_data_file, num_epochs=50)