import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

def plot_gaussian_field(nc_file, episode=0, stop_pos=None, traj_xy=None, save_path='gaussian_field.png'):
    with Dataset(nc_file) as nc:
        conc = nc['concentration'][episode][~np.isnan(nc['concentration'][episode])]
        x = nc['x'][episode][:len(conc)]
        y = nc['y'][episode][:len(conc)]
        sigma = nc['gaussian_sigma'][episode]
        peak = nc['peak_concentration'][episode]
        source_x = nc['source_x'][episode]
        source_y = nc['source_y'][episode]
    plt.figure(figsize=(10,6))
    # 路径连线
    if traj_xy is not None:
        plt.plot(traj_xy[:,0], traj_xy[:,1], color='deepskyblue', linewidth=2, label='Trajectory')
        plt.scatter(traj_xy[:,0], traj_xy[:,1], c='deepskyblue', s=10)
    else:
        plt.plot(x, y, color='deepskyblue', linewidth=2, label='Trajectory')
        plt.scatter(x, y, c=conc, cmap='viridis', s=10)
    plt.colorbar(label='Concentration')
    plt.plot(source_x, source_y, 'ro', markersize=10, label='Source')
    if stop_pos is not None:
        plt.plot(stop_pos[0], stop_pos[1], 'ws', markersize=12, markeredgecolor='k', label='Stop')
    plt.title(f"σ={sigma:.1f}m, Peak={peak:.1f}ppm")
    plt.legend()
    plt.savefig(save_path)
    plt.close()