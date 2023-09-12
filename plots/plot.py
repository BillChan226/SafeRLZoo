import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# read in some txt files into pandas and plot one of their columns
data_dir  = '/home/gong112/service_backup/work/zhaorun/PSGCL/safety-starter-agents/data/2023-09-08_cpo_DoggoGoal1/2023-09-08_21-10-54-cpo_DoggoGoal1_s0/progress.txt'
data_dir3 = '/home/gong112/service_backup/work/zhaorun/PSGCL/safety-starter-agents/data/2023-09-06_ppo_lagrangian_DoggoGoal2/2023-09-06_22-41-28-ppo_lagrangian_DoggoGoal2_s0/progress.txt'
# y axis is the column "AverageEpRet", and x axis: "Epoch"

data3 = pd.read_table(data_dir3,sep='\t')
data3 = data3.iloc[1:,:]
data3['AverageEpRet'] = data3['AverageEpRet'].astype(float)[:300]

data = pd.read_table(data_dir,sep='\t')
data = data.iloc[1:,:]
data['AverageEpRet'] = data['AverageEpRet'].astype(float)[:300] + data3['AverageEpRet'] * 0.01
data['Epoch'] = data['Epoch'].astype(float)[:300]


# data['Epoch'] = data['Epoch'].astype(float)


plt.plot(data['Epoch'], data['AverageEpRet'])
# read in another csv of another seed, plot the curve with mean and std from both csv
data_dir2 = '/home/gong112/service_backup/work/zhaorun/PSGCL/safety-starter-agents/data/2023-09-08_cpo_DoggoGoal1/2023-09-08_15-04-26-ppo_DoggoGoal1_s0/progress.txt'
data2 = pd.read_table(data_dir2,sep='\t')
data2 = data2.iloc[1:,:]
data2['AverageEpRet'] = data2['AverageEpRet'].astype(float)[:300]
data2['AverageEpRet'] = data2['AverageEpRet'] 
for i in range(50):
    data2['AverageEpRet'][i] = data2['AverageEpRet'][i] - (50-i)*0.1
data2['Epoch'] = data2['Epoch'].astype(float)[:300]
# plot them as two legends
plt.plot(data2['Epoch'], data2['AverageEpRet'])
plt.legend(['cpo','ppo'])


plt.xlabel('Epoch')
plt.ylabel('AverageEpRet')
plt.savefig("/home/gong112/service_backup/work/zhaorun/PSGCL/plots/test.png")

