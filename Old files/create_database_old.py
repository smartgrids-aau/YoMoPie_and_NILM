# importing the required module 
import matplotlib.pyplot as plt
import pandas as pd
import csv

# import time to measure the computation time required
from datetime import datetime
# read data vom csv file

start = datetime.now()

filename = "10Hz_testsample.csv"
df = pd.read_csv(filename)

# print whole dataframe
# print(df)

# create quantiles to remove outliers from the dataset
q_high = df["apparent.1"].quantile(0.975)
q_low = df["apparent.1"].quantile(0.025)

# create a new filtered dataset where outliers are removed
df_filtered = df[(df["apparent.1"] < q_high)]

def get_mean_apparent_power(index, number_of_values):
    mean = df_filtered['apparent.1'].iloc[[index-number_of_values/2, index+number_of_values/2]].mean(axis=0)
    return mean

def calculate_diff(index):
    diff = df_filtered['apparent.1'].iloc[index+1]-df_filtered['apparent.1'].iloc[index];
    return diff

def calculate_state_diff(index):
    diff = df_filtered['apparent_mean'].iloc[index+10]-df_filtered['apparent_mean'].iloc[index];
    return diff

def diff(array):
    difference = []
    difference.append(0)
    for i in range(0, len(array)-1):
        difference.append(array.iloc[i+1]-array.iloc[i])
    return difference 
# calculate mean apparent power with variable length of data (mean_size)
mean_size = 5
apparent_mean = []
for i in range(0,len(df_filtered)-mean_size):
    apparent_mean.append(get_mean_apparent_power(i,mean_size))
for i in range(len(df_filtered)-mean_size, len(df_filtered)):
    apparent_mean.append(get_mean_apparent_power(i, len(df_filtered)-i-1))
df_filtered['apparent_mean']=apparent_mean

# calculate the difference in apparent power mean
threshold = 5
difference = []
difference.append(0)
for i in range(0, len(df_filtered)-1):
    difference.append(calculate_diff(i))   
df_filtered['difference']=difference
df_filtered.difference[abs(df_filtered.difference)<threshold]=0
      
end = datetime.now()
print(end-start)

device1_on = df_filtered.loc[900:950,['apparent.1']]
print(device1_on.to_numpy().reshape(51))

# x = df_filtered.loc[1:,['t']]
# x = range(0,len(df_filtered))
# y = df_filtered.loc[:,['apparent.1']]
# plt.figure(1)
# plt.plot(x,y,'r')
# plt.title('Testsample full')
# plt.xlabel('Value')
# plt.ylabel('Apparent Power')
# 
# # device 1
# device1_on = df_filtered.loc[900:950,['apparent.1']]
# device1_on = diff(device1_on)    
# x_device1_on = range(0,len(device1_on))
# plt.figure(2)
# plt.plot(x_device1_on,device1_on,'r')
# plt.title('Device 1 Off->On')
# plt.xlabel('Value')
# plt.ylabel('Apparent Power')
# 
# device1_off = df_filtered.loc[1230:1280,['apparent.1']]
# device1_off = diff(device1_off)    
# x_device1_off = range(0,len(device1_off))
# plt.figure(3)
# plt.plot(x_device1_off,device1_off,'r')
# plt.title('Device 1 On->Off')
# plt.xlabel('Value')
# plt.ylabel('Apparent Power')
# 
# # device 2
# device2_on = df_filtered.loc[1450:1500,['apparent.1']]
# device2_on = diff(device2_on)    
# x_device2_on = range(0,len(device2_on))
# plt.figure(4)
# plt.plot(x_device2_on,device2_on,'r')
# plt.title('Device 2 Off->On')
# plt.xlabel('Value')
# plt.ylabel('Apparent Power')
# 
# device2_off = df_filtered.loc[1920:1970,['apparent.1']]
# device2_off = diff(device2_off)    
# x_device2_off = range(0,len(device2_off))
# plt.figure(5)
# plt.plot(x_device2_off,device2_off,'r')
# plt.title('Device 2 On->Off')
# plt.xlabel('Value')
# plt.ylabel('Apparent Power')
# 
# # device 3
# device3_on = df_filtered.loc[1966:2016,['apparent.1']]
# device3_on = diff(device3_on)    
# x_device3_on = range(0,len(device3_on))
# plt.figure(6)
# plt.plot(x_device3_on,device3_on,'r')
# plt.title('Device 3 Off->On')
# plt.xlabel('Value')
# plt.ylabel('Apparent Power')
# 
# device3_off = df_filtered.loc[2400:2450,['apparent.1']]
# device3_off = diff(device3_off)    
# x_device3_off = range(0,len(device3_off))
# plt.figure(7)
# plt.plot(x_device3_off,device3_off,'r')
# plt.title('Device 3 On->Off')
# plt.xlabel('Value')
# plt.ylabel('Apparent Power')
# 
# # device 4
# device4_on = df_filtered.loc[3590:3640,['apparent.1']]
# device4_on = diff(device4_on)    
# x_device4_on = range(0,len(device4_on))
# plt.figure(8)
# plt.plot(x_device4_on,device4_on,'r')
# plt.title('Device 4 Off->On')
# plt.xlabel('Value')
# plt.ylabel('Apparent Power')
# 
# device4_off = df_filtered.loc[4400:4450,['apparent.1']]
# device4_off = diff(device4_off)    
# x_device4_off = range(0,len(device4_off))
# plt.figure(9)
# plt.plot(x_device4_off,device4_off,'r')
# plt.title('Device 4 On->Off')
# plt.xlabel('Value')
# plt.ylabel('Apparent Power')
# 
# print(device1_on)
# df = pd.DataFrame(device1_on, columns=['apparent'])
# df.to_csv('device1_on.csv')
# 
# plt.show()


