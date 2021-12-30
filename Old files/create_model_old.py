import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

# load the data
filename = "10Hz_testsample.csv"
df = pd.read_csv(filename)

def diff(array):
    difference = []
    difference.append(0)
    for i in range(0, len(array)-1):
        difference.append(array[i+1]-array[i])
    return difference

def remove_outliers(array):
    array_filtered = []
    for i in range(0, len(array)-1):
        if abs(array[i])>3600:
            array_filtered.append(0)
        else:
            array_filtered.append(array[i])
    return array_filtered
# create training data
device1_on = df.loc[900:950,['apparent.1']].to_numpy().reshape(51)
device1_on = np.array([remove_outliers(diff(device1_on))])
device1_off = df.loc[1230:1280,['apparent.1']].to_numpy().reshape(51)
device1_off = np.array([remove_outliers(diff(device1_off))])
device2_on = df.loc[1450:1500,['apparent.1']].to_numpy().reshape(51)
device2_on = np.array([remove_outliers(diff(device2_on))])
device2_off = df.loc[1920:1970,['apparent.1']].to_numpy().reshape(51)
device2_off = np.array([remove_outliers(diff(device2_off))])
device3_on = df.loc[1966:2016,['apparent.1']].to_numpy().reshape(51)
device3_on = np.array([remove_outliers(diff(device3_on))])
device3_off = df.loc[2400:2450,['apparent.1']].to_numpy().reshape(51)
device3_off = np.array([remove_outliers(diff(device3_off))])
device4_on = df.loc[3590:3640,['apparent.1']].to_numpy().reshape(51)
device4_on = np.array([remove_outliers(diff(device4_on))])
device4_off = df.loc[4400:4450,['apparent.1']].to_numpy().reshape(51)
device4_off = np.array([remove_outliers(diff(device4_off))])
 
training_data = np.array([device1_on, device1_off, device2_on, device2_off, device3_on, device3_off, device4_on, device4_off])
training_labels = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# define the model structure
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(training_data, training_labels, epochs=100)

# save whole model with weights in this file
model.save("first_model")

# load the whole model with weights in this file
model = tf.keras.models.load_model("first_model")

# test the model with a 51 sample window

result = np.array([1])
result_full = np.array([1])
for i in range(0, len(df)-50, 10):
    test_device1 = df.loc[i:i+50,['apparent.1']].to_numpy().reshape(51)
    test_device1 = np.array([remove_outliers(diff(test_device1))])
    prediction = model.predict(test_device1)
    result = np.append(result, [np.argmax(prediction)])
    for j in range (0,9):
        result_full = np.append(result_full, [np.argmax(prediction)])
    #print(np.argmax(prediction))

x = range(0,len(result_full))
plt.plot(x,result_full, 'r')
plt.show()