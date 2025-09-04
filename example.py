import sys
import os

# 添加源代码路径到 Python 搜索路径
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from raytrace2d.env import Interpolator
from raytrace2d.raytrace import SoundSpeedProfile
from raytrace2d.plotting import plot_bathymetry

# 构造距离和水深
distance = np.linspace(0, 1000, 100)
water_depth = 100 * np.ones_like(distance)

# 构造声速剖面（简单线性变化）
depths = np.linspace(0, 100, 50)
sound_speeds = 1500 + 0.017 * depths  # 线性变化

# 创建声速插值器 - 注意 points 参数需要是元组格式
ssp = Interpolator(points=(depths,), values=sound_speeds)

# 绘制水深和声速剖面
fig, ax = plt.subplots()
plot_bathymetry(distance, water_depth, ax=ax)
ax.set_xlabel("距离 (m)")
ax.set_ylabel("水深 (m)")
ax.set_title("水深剖面")
plt.show()

# 更多高级用法可参考 raytrace2d.raytrace 里的方法，结合实际场景设置源、接收器等参数