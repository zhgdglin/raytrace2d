import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# 添加源代码路径到 Python 搜索路径
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# 检查并安装缺失的依赖
try:
    import scipy
    print("✓ scipy 可用")
except ImportError:
    print("❌ 需要安装 scipy: pip install scipy")
    exit(1)

from raytrace2d.env import Interpolator, Bathymetry, Source, Receiver
from raytrace2d.raytrace import RayTrace
import raytrace2d.plotting as rplt
from raytrace2d.utils import douglas_peucker

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SoundSpeedProfile:
    """声速剖面类"""
    def __init__(self, depths, speeds):
        self.depths = depths
        self.speeds = speeds
        self.interpolator = Interpolator(points=(depths,), values=speeds)
        
        # 计算声速梯度
        self.speed_gradient = np.gradient(speeds, depths)
        self.gradient_interpolator = Interpolator(points=(depths,), values=self.speed_gradient)
    
    def __call__(self, depth, distance=None):
        """返回给定深度的声速"""
        if distance is not None:
            # 2D 情况，暂时忽略水平变化
            pass
        # 返回慢度 (slowness) = 1 / c(z)
        c = self.interpolator(depth)
        return 1.0 / c

    def speed(self, depth):
        """返回速度 c(z)"""
        return self.interpolator(depth)
    
    def slowness_gradient(self, depth, distance=None):
        """返回慢度梯度 (dslowness/dz, dslowness/dx)"""
        c = self.speed(depth)
        dc_dz = self.gradient_interpolator(depth)
        # s = 1/c, ds/dz = -(1/c^2) * dc/dz
        dslowness_dz = -dc_dz / (c ** 2)
        dslowness_dx = 0.0  # 假设水平方向无变化
        return dslowness_dz, dslowness_dx

def horizontal_distance_from_path(ray, profile, stop_index: Optional[int] = None, eps: float = 1e-9) -> float:
    """根据给定公式由传播路径反推出水平距离。

    x = Σ_{i=0}^{N-2} [ c(z_i) / (g_i * cos(theta_i)) ] * |sin(theta_i) - sin(theta_{i+1})|

    其中：
    - c(z_i): 深度 z_i 处的声速
    - g_i: 声速梯度 dc/dz 在 z_i 处
    - theta_i: 射线在第 i 个步上的切向角（相对水平）
    - 本实现直接使用 ray.tang 为角度序列，z 为深度序列。
    """
    import numpy as np

    theta = np.asarray(ray.tang)
    z = np.asarray(ray.z)
    # 只累加到指定索引（通常是最接近接收器的点），避免把后续段落也计入
    if stop_index is not None and stop_index > 0:
        theta = theta[: stop_index + 1]
        z = z[: stop_index + 1]
    if theta.size < 2:
        return 0.0

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # 在各 z_i 处取声速和梯度（到倒数第二项，因为用到 i+1）
    z_i = z[:-1]
    # 速度 c(z)
    c_i = profile.speed(z_i)
    # 声速梯度 dc/dz：优先用 profile.gradient_interpolator；否则用差分近似
    if hasattr(profile, "gradient_interpolator") and profile.gradient_interpolator is not None:
        g_i = profile.gradient_interpolator(z_i)
    else:
        c_all = profile.speed(z)
        # 用 z 作为自变量做数值梯度
        g_all = np.gradient(c_all, z, edge_order=2)
        g_i = g_all[:-1]

    # 计算每一段的增量；避免除零
    denom = g_i * cos_t[:-1]
    denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps + (denom == 0) * eps, denom)
    increments = (c_i / denom) * np.abs(sin_t[:-1] - sin_t[1:])
    # 过滤 NaN/Inf
    increments = np.where(np.isfinite(increments), increments, 0.0)
    return float(np.sum(increments))

def load_sound_speed_data(filename):
    """加载声速剖面数据"""
    print(f"正在加载声速数据: {filename}")
    data = np.loadtxt(filename, delimiter=',', usecols=(0, 1))
    depths = data[:, 0]
    speeds = data[:, 1]
    
    print(f"数据点数: {len(depths)}")
    print(f"深度范围: {depths.min():.1f} - {depths.max():.1f} m")
    print(f"声速范围: {speeds.min():.1f} - {speeds.max():.1f} m/s")
    
    return depths, speeds

def create_ray_trace_example():
    """创建声线追踪示例"""
    
    # 1. 加载声速剖面数据
    depths, speeds = load_sound_speed_data('matrix_data.csv')
    
    # 创建声速剖面对象
    # 步骤1：简化声速剖面
    epsilon = 5.0  # 阈值 T，可根据需要调整
    depths_s, speeds_s = douglas_peucker(depths, speeds, epsilon)
    print(f"简化前点数: {len(depths)}, 简化后点数: {len(depths_s)}")
    # 可视化原始与简化剖面对比
    plt.figure(figsize=(6, 4))
    plt.plot(speeds, depths, 'o-', label='原始剖面', alpha=0.6)
    plt.plot(speeds_s, depths_s, 's-', label='简化剖面', alpha=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel('声速 (m/s)')
    plt.ylabel('深度 (m)')
    plt.title(f'声速剖面抽稀 (epsilon={epsilon})')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # 使用简化后的剖面创建对象
    profile = SoundSpeedProfile(depths_s, speeds_s)
    
    # 2. 定义海底地形（平底）
    max_distance = 5000  # 5km 距离
    max_depth = depths.max()
    
    # 简单平底地形
    bathymetry = Bathymetry(
        water_depth=np.array([max_depth * 0.9, max_depth * 0.9]),  # 必须是数组
        distance=np.array([0, max_distance])
    )
    
    # 3. 定义声源和接收器
    source = Source(
        depth=100.0,        # 声源深度 100m
        distance=0.0        # 声源位置 0m
    )
    
    receiver = Receiver(
        depth=1000.0,        # 接收器深度 200m  
        distance=3000.0     # 接收器距离 3000m
    )
    
    print(f"\n设置参数:")
    print(f"声源位置: 深度 {source.depth}m, 距离 {source.distance}m")
    print(f"接收器位置: 深度 {receiver.depth}m, 距离 {receiver.distance}m")
    print(f"海底深度: {bathymetry.water_depth}m")
    
    # 4. 创建射线追踪对象
    raytrace = RayTrace(
        profile=profile,
        bathymetry=bathymetry,
        source=source,
        receiver=receiver
    )
    
    # 5. 计算射线路径
    print("\n正在计算射线路径...")
    
    # 发射多条不同角度的射线
    launch_angles = np.linspace(-30, 30, 21)  # -30° 到 +30°，21条射线
    
    # 使用 run 方法计算射线和特征射线
    rays, eigenrays = raytrace.run(
        angles=launch_angles,
        ds=100.0,           # 步长 50m
        eigenrays=True,    # 计算特征射线
        num_workers=4      # 并行计算
    )
    
    print(f"成功计算 {len(rays)} 条射线")
    if eigenrays:
        print(f"找到 {len(eigenrays)} 条特征射线")
        print("\n特征射线详情：")
        for j, er in enumerate(eigenrays):
            # 特征射线到达接收器处的指标与误差
            print(
                f"  [ER{j}] 路径类型={er.path_phase}, 发射角={er.launch_angle:.2f}°, "
                f"传播时间={er.tau[-1]:.4f}s, 反射次数={er.num_reflections}, 深度误差={er.depth_error:.3f} m"
            )
    else:
        print("未找到特征射线")

    # 仅计算并打印特征射线的水平传播距离（基于路径反推公式）
    if eigenrays:
        print("\n特征射线水平传播距离与传播时间（在接收器处）:")
        for j, er in enumerate(eigenrays):
            # 找到最接近接收器水平位置的索引
            idx = int(np.argmin(np.abs(er.x - raytrace.receiver.distance)))
            # 在该位置的传播时间
            t_rcv = float(er.tau[idx])
            # 基于路径反推的水平距离（只累计到接收器位置索引）
            hd = horizontal_distance_from_path(er, profile, stop_index=idx)
            print(f"  [ER{j}] 路径类型={er.path_phase}, 水平距离={hd:.2f} m, 传播时间={t_rcv:.4f} s")
    
    # 6. 绘制结果（含原始、简化声速剖面和射线路径）
    plot_results(raytrace, depths, speeds, depths_s, speeds_s)
    # 仅绘制特征射线的传播路径
    plot_eigenrays_only(raytrace, depths)
    
    return raytrace

def plot_results(raytrace, depths, speeds, depths_s, speeds_s):
    """绘制射线追踪结果，包括原始、简化声速剖面和射线路径"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 绘制原始声速剖面
    ax1.plot(speeds, depths, 'b-', linewidth=2, label='原始声速剖面')
    ax1.set_xlabel('声速 (m/s)')
    ax1.set_ylabel('深度 (m)')
    ax1.set_title('声速剖面')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 绘制射线路径
    for i, ray in enumerate(raytrace.rays):
        color = plt.cm.viridis(i / len(raytrace.rays))
        ax2.plot(ray.x, ray.z, color=color, alpha=0.7, linewidth=1)
    
    # 标记声源和接收器
    ax2.plot(raytrace.source.distance, raytrace.source.depth, 
             'r*', markersize=15, label='声源')
    ax2.plot(raytrace.receiver.distance, raytrace.receiver.depth, 
             'ro', markersize=10, label='接收器')
    
    # ...existing海底绘制代码...
    x_bottom = np.linspace(0, 5000, 100)
    y_bottom = raytrace.bathymetry(x_bottom)
    ax2.fill_between(x_bottom, y_bottom, np.max(depths), 
                     color='brown', alpha=0.3, label='海底')
    
    ax2.set_xlabel('距离 (m)')
    ax2.set_ylabel('深度 (m)')
    ax2.set_title('声线传播路径')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 绘制简化声速剖面
    ax3.plot(speeds_s, depths_s, 'g-o', linewidth=2, markersize=4, label='简化声速剖面')
    ax3.set_xlabel('声速 (m/s)')
    ax3.set_ylabel('深度 (m)')
    ax3.set_title('简化声速剖面')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()
    
    print("✓ 绘制完成！")

def plot_eigenrays_only(raytrace, depths):
    """仅绘制特征射线的传播路径。如果没有特征射线，将打印提示。"""
    if not raytrace.eigenrays:
        print("无特征射线可绘制。")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # 绘制特征射线
    for i, er in enumerate(raytrace.eigenrays):
        color = plt.cm.tab10(i % 10)
        ax.plot(er.x, er.z, color=color, linewidth=2, label=f'ER{i} ({er.path_phase})')

    # 标记声源和接收器
    ax.plot(raytrace.source.distance, raytrace.source.depth, 'r*', markersize=14, label='声源')
    ax.plot(raytrace.receiver.distance, raytrace.receiver.depth, 'ro', markersize=8, label='接收器')

    # 绘制海底
    x_bottom = np.linspace(0, max(er.x.max() for er in raytrace.eigenrays), 200)
    y_bottom = raytrace.bathymetry(x_bottom)
    ax.fill_between(x_bottom, y_bottom, np.max(depths), color='brown', alpha=0.25, label='海底')

    ax.set_xlabel('距离 (m)')
    ax.set_ylabel('深度 (m)')
    ax.set_title('特征射线传播路径')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        raytrace = create_ray_trace_example()
        print("\n✓ 声线追踪计算成功完成！")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("请检查:")
        print("1. matrix_data.csv 文件是否存在")
        print("2. 是否安装了所有依赖: pip install scipy")
        print("3. Python 版本兼容性（建议 3.10+）")
