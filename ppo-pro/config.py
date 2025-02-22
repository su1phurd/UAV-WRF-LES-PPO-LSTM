# config.py

# ======================
# 超参数配置
# ======================
GRID_SIZE = 500        # 区域尺寸
MAX_STEPS = 1000       # 最大步数
CONC_PEAK = 100.0      # 峰值浓度
TURBULENCE_INTENSITY = 3.0  # 湍流强度

# PPO参数
GAMMA = 0.99  #未来奖励折扣因子
LAMBDA = 0.95 #GAE参数（平衡TD与MC）
CLIP_EPSILON = 0.2 #策略更新剪切范围
ENTROPY_BETA = 0.01 #熵正则化系数
LEARNING_RATE = 1e-4 #学习率
BATCH_SIZE = 256 #每次更新使用的样本量
EPOCHS = 5 #每次采样的训练轮次

# 探索参数
EXPLORE_BONUS = 1.0     # 基础探索奖励
DECAY_FACTOR = 0.995    # 探索衰减
GRID_DIVISIONS = 10     # 探索网格划分

# 课程学习参数
INITIAL_RADIUS = 50.0 #初始目标判定半径
MIN_RADIUS = 5.0 #初始目标判定半径
RADIUS_DECAY = 0.95 #半径衰减系数
SUCCESS_THRESHOLD = 0.7 #触发收缩的成功率阈值
WINDOW_SIZE = 200 #成功率计算窗口