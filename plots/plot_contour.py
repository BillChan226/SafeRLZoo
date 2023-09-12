import numpy as np
import torch
import matplotlib.pyplot as plt


env_space = np.arange(-3, 3, 0.01)
env_matrix = []
# for i in env_space:
#     for j in env_space:
#         env_matrix.append([i,j])
# env_matrix = np.array(env_matrix)
# print("env_matrix", env_matrix)
# env_matrix.reshape(6, 6, 2)
# print("env_matrix", env_matrix)

# x, y = np.meshgrid(env_space, env_space)
# print("mesh", x)    
# print("mesh", y)   
# read in a csv file
# v_matrix = np.loadtxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/v_matrix50.csv", delimiter=",")
# v_safe_matrix = np.loadtxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/v_safe_matrix30.csv", delimiter=",")
row = 1000

ori_safe_matrix = np.loadtxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour3/v_safe_matrix10.csv", delimiter=",")
env_space = np.arange(-3, 3, 0.02)
std_safe_matrix = np.loadtxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour4/v_safe_matrix100.csv", delimiter=",")
# v_matrix = v_matrix[180:780, 270:870]
allones= np.loadtxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour4/v_matrix100.csv", delimiter=",")

# np.savetxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/v_matrix.csv", v_matrix, delimiter=",")
# np.savetxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/v_safe_matrix.csv", v_safe_matrix, delimiter=",")
# transfer ori_safe_matrix from (1200,1200) to (300,300)

v_matrix = std_safe_matrix - 0.3* ori_safe_matrix[::4, ::4] + 1* allones
np.savetxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/std_matrix.csv", v_matrix, delimiter=",")

# print("v_matrix", np.shape(v_matrix))
# print("v_safe_matrix", np.shape(v_safe_matrix))
plt.contourf(env_space, env_space, v_matrix)
C = plt.contour(env_space, env_space, v_matrix, colors='black', linewidths=0.8)
plt.clabel(C, inline=True, fontsize=12)
plt.savefig("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/contour_safe_std.png")

# plt.savefig("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/contour_goal.png")
plt.clf()
# env_space = np.arange(-3, 3, 0.005)
# row = 1200
# plt.contourf(env_space, env_space, v_safe_matrix.reshape(row,row))
# C = plt.contour(env_space, env_space, v_safe_matrix.reshape(row,row), colors='black', linewidths=0.8)
# plt.clabel(C, inline=True, fontsize=12)
# plt.savefig("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/contour_safety.png")
# plt.clf()
    
    
    
    # plt.contourf(env_space, env_space, v_matrix.reshape(1000,1000))
    # C = plt.contour(env_space, env_space, v_matrix.reshape(1000,1000), colors='black', linewidths=0.8)
    # plt.clabel(C, inline=True, fontsize=12)
    # plt.savefig("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contourf.png")