# importing the required module 
import matplotlib.pyplot as plt
import pandas as pd

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
    diff = df_filtered['apparent_mean'].iloc[index+1]-df_filtered['apparent_mean'].iloc[index];
    return diff

def calculate_state_diff(index):
    diff = df_filtered['apparent_mean'].iloc[index+10]-df_filtered['apparent_mean'].iloc[index];
    return diff
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
#print(df_filtered)
# x = df_filtered.loc[1:,['t']]
x = range(0,len(df_filtered))
y = df_filtered.loc[:,['apparent.1']]
plt.plot(x,y, 'r')
plt.xlabel('Value')
plt.ylabel('Apparent Power')
plt.title('Apparent Power over Time (97.5th percentile)')
y_mean = df_filtered.loc[:,['apparent_mean']]
plt.plot(x,y_mean, 'b')
y_diff = df_filtered.loc[:,['difference']]
plt.plot(x,y_diff, 'g')
plt.show()
