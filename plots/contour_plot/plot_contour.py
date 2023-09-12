import numpy as np
import torch
import matplotlib.pyplot as plt


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
v_matrix = np.loadtxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/v_matrix.csv", delimiter=",")
v_safe_matrix = np.loadtxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/v_safe_matrix.csv", delimiter=",")
std_safe_matrix = np.loadtxt("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/std_matrix.csv", delimiter=",")



print("v_matrix", np.shape(v_matrix))
print("v_safe_matrix", np.shape(v_safe_matrix))
print("std_safe_matrix", np.shape(std_safe_matrix))
env_space = np.arange(-3, 3, 0.01)
plt.contourf(env_space, env_space, v_matrix, cmap="hot")
C = plt.contour(env_space, env_space, v_matrix, colors='black', linewidths=0.8)
plt.clabel(C, inline=True, fontsize=12)

plt.savefig("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/contour_goal.png")
plt.clf()
env_space = np.arange(-3, 3, 0.005)
row = 1200
plt.contourf(env_space, env_space, v_safe_matrix)
C = plt.contour(env_space, env_space, v_safe_matrix, colors='black', linewidths=0.8)
plt.clabel(C, inline=True, fontsize=12)
plt.savefig("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/contour_safety.png")
plt.clf()

env_space = np.arange(-3, 3, 0.02)
plt.contourf(env_space, env_space, std_safe_matrix)
C = plt.contour(env_space, env_space, std_safe_matrix, colors='black', linewidths=0.8)
plt.clabel(C, inline=True, fontsize=12)
plt.savefig("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/contour_plot/contour_safe_std.png")
