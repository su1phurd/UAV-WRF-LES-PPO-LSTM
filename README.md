<div align="center">
  <h1>无人机自主溯源甲烷羽流系统<br>Autonomous UAV Methane Plume Tracing System</h1>
  
  [中文版本](#chinese) | [English Version](#english)  
  ![GitHub](https://img.shields.io/badge/Algorithm-PPO%2BLSTM-blue) 
  ![GitHub](https://img.shields.io/badge/Platform-ROS%2FGazebo-green)
  ![GitHub](https://img.shields.io/badge/Language-Python%2FFortran-orange)
</div>

---

## <span id="chinese">中文版本</span> [↑返回顶部](#top)

### 项目背景
南京大学大学生创新计划项目，开发基于**强化学习**的无人机自主导航系统，用于：
1. 追踪大气甲烷羽流至高斯中心
2. 定位点源位置（误差<5米）
3. 反演排放通量（误差<20%）

### 算法版本
#### PPO 1.0（基础版）
```text
├── ppo1.0/
│   ├── ppo_basic/       # 标准PPO算法实现
│   ├── fixed_threshold/ # 经验浓度阈值（800-1200ppb停止）
│   └── gaussian_env/    # 高斯羽流仿真环境
```
- **特点**：首次实现PPO与化学阈值停止的融合

#### PPO 2.0（LSTM增强版）
```text
├── ppo2.0/
│   ├── lstm_module/     # 浓度时间序列预测器
│   ├── dynamic_stop/    # 动态停止阈值（500-1500ppb）
│   └── nc_analyzer/     # 分析训练输出的阈值优化
```
- **改进**：LSTM预测最优停止阈值（测试集R²=0.82）

#### PPO 2.1（趋势分析版）
```text
├── ppo2.1/
│   ├── gradient_detec/  # 基于∇[CH₄]的源定位
│   └── trend_predict/   # 通过dC/dt模式确认源区
```
- **突破**：完全摒弃固定阈值，采用微分趋势分析

### 技术参数
| 组件               | 实现细节                          |
|--------------------|-----------------------------------|
| 羽流模型           | 高斯扩散模型（σ_y=0.3x^0.71）     |
| 状态空间           | [CH₄]、风速矢量、无人机位置       |
| 奖励函数           | R = Δ[CH₄] - 0.2‖Δθ‖             |
| 训练硬件           | NVIDIA RTX 3090（3840 CUDA核心）  |

[↑返回顶部](#top)

---

## <span id="english">English Version</span> [↑Back to Top](#top)

### Project Background
Nanjing University Innovation Program developing **reinforcement learning** UAV system for:
1. Tracing methane plumes to Gaussian centers
2. Locating point sources (<5m error)
3. Quantifying emission fluxes (<20% error)

### Algorithm Versions
#### PPO 1.0 (Baseline)
```text
├── ppo1.0/
│   ├── ppo_basic/       # Standard PPO implementation  
│   ├── fixed_threshold/ # Empirical stop threshold (800-1200ppb)
│   └── gaussian_env/    # Gaussian plume simulation
```
- **Key Feature**: First integration of PPO with chemical threshold stopping

#### PPO 2.0 (LSTM-enhanced)
```text
├── ppo2.0/
│   ├── lstm_module/     # Concentration time-series predictor
│   ├── dynamic_stop/    # Adaptive stopping threshold (500-1500ppb)  
│   └── nc_analyzer/     # Threshold optimization from training outputs
```
- **Improvement**: LSTM predicts optimal stop threshold (R²=0.82)

#### PPO 2.1 (Trend-based)
```text
├── ppo2.1/  
│   ├── gradient_detec/  # Source localization via ∇[CH₄]
│   └── trend_predict/   # Source confirmation through dC/dt patterns
```
- **Breakthrough**: Eliminates fixed thresholds using derivative analysis

### Technical Specifications
| Component         | Implementation Details               |
|-------------------|--------------------------------------|
| Plume Model       | Gaussian dispersion (σ_y=0.3x^0.71)  |
| State Space       | [CH₄], wind vector, UAV position     |
| Reward Function   | R = Δ[CH₄] - 0.2‖Δθ‖                |
| Training Hardware | NVIDIA RTX 3090 (3840 CUDA cores)    |

[↑Back to Top](#top)

---

<div align="center">
  :memo: <strong>Citation 引用格式</strong><br>
  Nanjing University CH₄ UAV Team. (2023). Autonomous Plume Tracing System. <i>Student Innovation Program</i>.
</div>
```
