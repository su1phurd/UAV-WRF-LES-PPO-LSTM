import numpy as np
from netCDF4 import Dataset

class NetCDFWriter:
    def __init__(self, filename, grid_size, max_episodes=2000, max_steps=1000):
        """
        初始化 NetCDF 文件
        :param filename: 输出文件名
        :param grid_size: 环境网格大小
        :param max_episodes: 最大训练轮次
        :param max_steps: 每轮最大步数
        """
        self.filename = filename
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        
        # 创建文件并定义维度
        self.ncfile = Dataset(filename, mode='w', format='NETCDF4')
        self.ncfile.createDimension('episode', max_episodes)
        self.ncfile.createDimension('step', max_steps)
        
        # 定义全局属性
        self.ncfile.GRID_SIZE = grid_size
        
        # 定义变量
        self._init_variables()
    
    def _init_variables(self):
        """定义所有变量及其元数据"""
        # Episode 编号 (0~1999)
        self.episode_var = self.ncfile.createVariable(
            'episode', np.int32, ('episode',)
        )
        self.episode_var.long_name = "Training episode index"
        
        # 步数编号 (0~999)
        self.step_var = self.ncfile.createVariable(
            'step', np.int32, ('step',)
        )
        self.step_var.long_name = "Step index within episode"
        
        # 坐标和浓度数据
        self.x_var = self.ncfile.createVariable(
            'x', np.float32, ('episode', 'step'),
            fill_value=np.nan, zlib=True
        )
        self.x_var.units = "grid unit"
        self.x_var.long_name = "Agent x-coordinate"
        
        self.y_var = self.ncfile.createVariable(
            'y', np.float32, ('episode', 'step'),
            fill_value=np.nan, zlib=True
        )
        self.y_var.units = "grid unit"
        self.y_var.long_name = "Agent y-coordinate"
        
        self.conc_var = self.ncfile.createVariable(
            'concentration', np.float32, ('episode', 'step'),
            fill_value=np.nan, zlib=True
        )
        self.conc_var.long_name = "Methane concentration"
        
        # 源头标记 (0-普通点, 1-源头)
        self.source_var = self.ncfile.createVariable(
            'is_source', np.int8, ('episode', 'step'),
            fill_value=0, zlib=True
        )
        self.source_var.long_name = "Source position flag"
        
        # 每个episode的源头浓度和实际坐标
        self.source_conc_var = self.ncfile.createVariable(
            'source_concentration', np.float32, ('episode',),
            fill_value=np.nan, zlib=True
        )
        self.source_conc_var.long_name = "Actual source concentration in each episode"
        
        self.source_x_var = self.ncfile.createVariable(
            'source_x', np.float32, ('episode',),
            fill_value=np.nan, zlib=True
        )
        self.source_x_var.long_name = "Actual source x-coordinate"
        
        self.source_y_var = self.ncfile.createVariable(
            'source_y', np.float32, ('episode',),
            fill_value=np.nan, zlib=True
        )
        self.source_y_var.long_name = "Actual source y-coordinate"
    
    def write_episode_data(self, episode_idx, steps, x, y, conc, source_x, source_y, source_conc):
        """
        写入单轮训练数据（仅在成功时调用）
        :param episode_idx: 轮次索引 (0~1999)
        :param steps: 实际步数
        :param x: x坐标数组 (长度=steps)
        :param y: y坐标数组 (长度=steps)
        :param conc: 浓度数组 (长度=steps)
        :param source_x: 实际到达的源头x坐标
        :param source_y: 实际到达的源头y坐标
        :param source_conc: 实际到达位置的浓度
        """
        # 填充步数数据
        self.x_var[episode_idx, :steps] = x
        self.y_var[episode_idx, :steps] = y
        self.conc_var[episode_idx, :steps] = conc
        
        # 标记源头位置（最后一步）
        self.source_var[episode_idx, steps-1] = 1
        self.x_var[episode_idx, steps-1] = source_x
        self.y_var[episode_idx, steps-1] = source_y
        
        # 记录源头坐标和浓度
        self.source_conc_var[episode_idx] = source_conc
        self.source_x_var[episode_idx] = source_x
        self.source_y_var[episode_idx] = source_y
    
    def close(self):
        """关闭文件"""
        self.ncfile.close()