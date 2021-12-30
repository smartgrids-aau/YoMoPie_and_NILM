import YoMoPie as yomopie
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import math
  
yomo = yomopie.YoMoPie()
device_states = [0,0,0,0,0,0,0]
# 1 - Fan
# 2 - Charger
# 3 - Hair dryer
# 4 - Lamp
# 5 - Toaster
# 6 - Vacuum
# 7 - Water kettle

def metering(sampleperiod, file):
    if (sampleperiod<0.1) or (file ==''):
        print('Incompatible sampling period or no file name!')
        return 1
    yomo.sample_intervall = sampleperiod
    yomo.disable_board()
    time.sleep(1)
    yomo.enable_board()
    time.sleep(1)
    while(1):
        sample = []
        sample.append(time.time())
        sample.append(yomo.get_apparent_energy()[1])
        print('%s' % sample[1])
        logfile = open(file,'a')
        for value in sample:
            logfile.write("%s;" % value)
        logfile.write("\n")
        time.sleep(sampleperiod);
    return 0

def metering_with_ld(file):
    if file =='':
        print('Unsupported filename!')
        return 1
    yomo.sample_intervall = 0.1
    yomo.disable_board()
    time.sleep(1)
    yomo.enable_board()
    time.sleep(1)
    step = 0
    cp = [0,0,0]
    samples = []
    while(1):
        step = step + 1
        sample = []
        sample.append(time.time())
        power=yomo.get_apparent_energy()[1]
        sample.append(power)
        samples.append(power)
        #print('%s' % sample[1])
        if step >= 10 and len(samples)>50:
            prediction = model.predict(np.array([diff(samples[len(samples)-50:len(samples)])]))
            print("----------------%f|%f----------------" % (np.argmax(prediction), np.amax(prediction)))
            if np.amax(prediction)>0.9:
                cp.append(np.argmax(prediction))
                if cp[len(cp)-1] == np.argmax(prediction) and cp[len(cp)-2] == np.argmax(prediction) and cp[len(cp)-3] == np.argmax(prediction):
                    change_state(np.argmax(prediction))
            step = 0
            print(device_states)
        logfile = open(file,'a')
        for value in sample:
            logfile.write("%s; " % value)
        logfile.write("\n")
        time.sleep(0.1);
    return 0

def change_state(state):
    if state%2==0:
        device_states[math.ceil(state/2)-1] = 0
    else:
        device_states[math.ceil(state/2)-1] = 1
    return 0

def diff(array):
    print("Diff: %f" %(array[len(array)-1]-array[0]))
    return array[len(array)-1]-array[0]

def diff2(array):
    difference = array[len(array)-1]-array[0]
    difference.append(0)
    for i in range(0, len(array)-1):
        difference.append(array[i+1]-array[i])
    return difference

def remove_outliers(array):
    array_filtered = array
    #for i in range(0, len(array)-1):
    #    if abs(array[i])>3600:
    #        array_filtered.append(0)
    #    else:
    #        array_filtered.append(array[i])
    return array_filtered

def create_model():
    # define the model structure
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(1),
                                        tf.keras.layers.Dense(64),
                                        tf.keras.layers.Dense(15, activation='softmax')])
    # compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Model compiled successful")
    return model

def train_model(training_data, training_labels, model):
    model.fit(training_data, training_labels, epochs=10000)
    return 0

def predict(filename):
    df = pd.read_csv(filename).loc[:,'apparent'].to_numpy()        
    result = np.array([0])
    for i in range(0, len(df)-50, 10):
        prediction = model.predict(np.array([diff(df[i:i+49])]))
        print("%f, %f" % (np.argmax(prediction), np.amax(prediction)))
        if np.amax(prediction)>0.9:
            result = np.append(result, [np.argmax(prediction)])
        else:
            result = np.append(result, [0])
    x = range(0,len(result))
    plt.figure(2)
    plt.plot(x,result, 'r')
    plt.show()
    return result

def predict2(filename):
    df = remove_outliers(pd.read_csv(filename).loc[:,'apparent'].to_numpy())
    x = range(0, len(df))
    plt.figure(1)
    plt.plot(x,df,'r')
    
    result = np.array([0])
    for i in range(0, len(df)-50, 10):
        prediction = model.predict(np.array([diff(df[i:i+49])]))
        print("%f, %f" % (np.argmax(prediction), np.amax(prediction)))
        if np.amax(prediction)>0.9:
            result = np.append(result, [np.argmax(prediction)])
        else:
            result = np.append(result, [0])
    x = range(0,len(result))
    plt.figure(2)
    plt.plot(x,result, 'r')
    plt.show()
    return result

# create a dataset with 50 samples and 10 hz
# yomo.disable_board()
# time.sleep(1)
# yomo.enable_board()
# time.sleep(1)
# model = tf.keras.models.load_model("test_model_1")
# #yomo.create_dataset(50,0.1, "test.csv")
# df = remove_outliers(diff(pd.read_csv("test.csv").loc[:,'apparent'].to_numpy()))
# x = range(0, len(df))
# plt.figure(1)
# plt.plot(x,df,'r')
# prediction = model.predict(np.array([remove_outliers(diff(pd.read_csv("test.csv").loc[:,'apparent'].to_numpy()))]))
# print(prediction)
# print("%f, %f" % (np.argmax(prediction), np.amax(prediction)))
# plt.figure(2)
# plt.plot([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],prediction[0],'r')
# plt.show()
# continuous metering of apparent power, 0.1s period
#yomo.disable_board()
#time.sleep(1)
#yomo.enable_board()
#time.sleep(1)
#metering(0.1, "smartmeter_data/data_test.csv")

'''
# Start of training process
# 1. read training data
# 2. create model
# 3. train model
# 4. save model local

# fan on
df_fan_on_1 = remove_outliers(diff(pd.read_csv("training_data/fan/fan_on_1.csv").loc[:,'apparent'].to_numpy()))
df_fan_on_2 = remove_outliers(diff(pd.read_csv("training_data/fan/fan_on_2.csv").loc[:,'apparent'].to_numpy()))
df_fan_on_3 = remove_outliers(diff(pd.read_csv("training_data/fan/fan_on_3.csv").loc[:,'apparent'].to_numpy()))
df_fan_on_4 = remove_outliers(diff(pd.read_csv("training_data/fan/fan_on_4.csv").loc[:,'apparent'].to_numpy()))
df_fan_on_5 = remove_outliers(diff(pd.read_csv("training_data/fan/fan_on_5.csv").loc[:,'apparent'].to_numpy()))
# fan off
df_fan_off_1 = remove_outliers(diff(pd.read_csv("training_data/fan/fan_off_1.csv").loc[:,'apparent'].to_numpy()))
df_fan_off_2 = remove_outliers(diff(pd.read_csv("training_data/fan/fan_off_2.csv").loc[:,'apparent'].to_numpy()))
df_fan_off_3 = remove_outliers(diff(pd.read_csv("training_data/fan/fan_off_3.csv").loc[:,'apparent'].to_numpy()))
df_fan_off_4 = remove_outliers(diff(pd.read_csv("training_data/fan/fan_off_4.csv").loc[:,'apparent'].to_numpy()))
df_fan_off_5 = remove_outliers(diff(pd.read_csv("training_data/fan/fan_off_5.csv").loc[:,'apparent'].to_numpy()))

# charger on
df_charger_on_1 = remove_outliers(diff(pd.read_csv("training_data/charger/charger_on_1.csv").loc[:,'apparent'].to_numpy()))
df_charger_on_2 = remove_outliers(diff(pd.read_csv("training_data/charger/charger_on_2.csv").loc[:,'apparent'].to_numpy()))
df_charger_on_3 = remove_outliers(diff(pd.read_csv("training_data/charger/charger_on_3.csv").loc[:,'apparent'].to_numpy()))
df_charger_on_4 = remove_outliers(diff(pd.read_csv("training_data/charger/charger_on_4.csv").loc[:,'apparent'].to_numpy()))
df_charger_on_5 = remove_outliers(diff(pd.read_csv("training_data/charger/charger_on_5.csv").loc[:,'apparent'].to_numpy()))
# charger off
df_charger_off_1 = remove_outliers(diff(pd.read_csv("training_data/charger/charger_off_1.csv").loc[:,'apparent'].to_numpy()))
df_charger_off_2 = remove_outliers(diff(pd.read_csv("training_data/charger/charger_off_2.csv").loc[:,'apparent'].to_numpy()))
df_charger_off_3 = remove_outliers(diff(pd.read_csv("training_data/charger/charger_off_3.csv").loc[:,'apparent'].to_numpy()))
df_charger_off_4 = remove_outliers(diff(pd.read_csv("training_data/charger/charger_off_4.csv").loc[:,'apparent'].to_numpy()))
df_charger_off_5 = remove_outliers(diff(pd.read_csv("training_data/charger/charger_off_5.csv").loc[:,'apparent'].to_numpy()))

# hairdryer on
df_hairdryer_on_1 = remove_outliers(diff(pd.read_csv("training_data/hairdryer/hairdryer_on_1.csv").loc[:,'apparent'].to_numpy()))
df_hairdryer_on_2 = remove_outliers(diff(pd.read_csv("training_data/hairdryer/hairdryer_on_2.csv").loc[:,'apparent'].to_numpy()))
df_hairdryer_on_3 = remove_outliers(diff(pd.read_csv("training_data/hairdryer/hairdryer_on_3.csv").loc[:,'apparent'].to_numpy()))
df_hairdryer_on_4 = remove_outliers(diff(pd.read_csv("training_data/hairdryer/hairdryer_on_4.csv").loc[:,'apparent'].to_numpy()))
df_hairdryer_on_5 = remove_outliers(diff(pd.read_csv("training_data/hairdryer/hairdryer_on_5.csv").loc[:,'apparent'].to_numpy()))
# hairdryer off
df_hairdryer_off_1 = remove_outliers(diff(pd.read_csv("training_data/hairdryer/hairdryer_off_1.csv").loc[:,'apparent'].to_numpy()))
df_hairdryer_off_2 = remove_outliers(diff(pd.read_csv("training_data/hairdryer/hairdryer_off_2.csv").loc[:,'apparent'].to_numpy()))
df_hairdryer_off_3 = remove_outliers(diff(pd.read_csv("training_data/hairdryer/hairdryer_off_3.csv").loc[:,'apparent'].to_numpy()))
df_hairdryer_off_4 = remove_outliers(diff(pd.read_csv("training_data/hairdryer/hairdryer_off_4.csv").loc[:,'apparent'].to_numpy()))
df_hairdryer_off_5 = remove_outliers(diff(pd.read_csv("training_data/hairdryer/hairdryer_off_5.csv").loc[:,'apparent'].to_numpy()))

# lamp on
df_lamp_on_1 = remove_outliers(diff(pd.read_csv("training_data/lamp/lamp_on_1.csv").loc[:,'apparent'].to_numpy()))
df_lamp_on_2 = remove_outliers(diff(pd.read_csv("training_data/lamp/lamp_on_2.csv").loc[:,'apparent'].to_numpy()))
df_lamp_on_3 = remove_outliers(diff(pd.read_csv("training_data/lamp/lamp_on_3.csv").loc[:,'apparent'].to_numpy()))
df_lamp_on_4 = remove_outliers(diff(pd.read_csv("training_data/lamp/lamp_on_4.csv").loc[:,'apparent'].to_numpy()))
df_lamp_on_5 = remove_outliers(diff(pd.read_csv("training_data/lamp/lamp_on_5.csv").loc[:,'apparent'].to_numpy()))
# lamp off
df_lamp_off_1 = remove_outliers(diff(pd.read_csv("training_data/lamp/lamp_off_1.csv").loc[:,'apparent'].to_numpy()))
df_lamp_off_2 = remove_outliers(diff(pd.read_csv("training_data/lamp/lamp_off_2.csv").loc[:,'apparent'].to_numpy()))
df_lamp_off_3 = remove_outliers(diff(pd.read_csv("training_data/lamp/lamp_off_3.csv").loc[:,'apparent'].to_numpy()))
df_lamp_off_4 = remove_outliers(diff(pd.read_csv("training_data/lamp/lamp_off_4.csv").loc[:,'apparent'].to_numpy()))
df_lamp_off_5 = remove_outliers(diff(pd.read_csv("training_data/lamp/lamp_off_5.csv").loc[:,'apparent'].to_numpy()))

# toaster on
df_toaster_on_1 = remove_outliers(diff(pd.read_csv("training_data/toaster/toaster_on_1.csv").loc[:,'apparent'].to_numpy()))
df_toaster_on_2 = remove_outliers(diff(pd.read_csv("training_data/toaster/toaster_on_2.csv").loc[:,'apparent'].to_numpy()))
df_toaster_on_3 = remove_outliers(diff(pd.read_csv("training_data/toaster/toaster_on_3.csv").loc[:,'apparent'].to_numpy()))
df_toaster_on_4 = remove_outliers(diff(pd.read_csv("training_data/toaster/toaster_on_4.csv").loc[:,'apparent'].to_numpy()))
df_toaster_on_5 = remove_outliers(diff(pd.read_csv("training_data/toaster/toaster_on_5.csv").loc[:,'apparent'].to_numpy()))
# toaster off
df_toaster_off_1 = remove_outliers(diff(pd.read_csv("training_data/toaster/toaster_off_1.csv").loc[:,'apparent'].to_numpy()))
df_toaster_off_2 = remove_outliers(diff(pd.read_csv("training_data/toaster/toaster_off_2.csv").loc[:,'apparent'].to_numpy()))
df_toaster_off_3 = remove_outliers(diff(pd.read_csv("training_data/toaster/toaster_off_3.csv").loc[:,'apparent'].to_numpy()))
df_toaster_off_4 = remove_outliers(diff(pd.read_csv("training_data/toaster/toaster_off_4.csv").loc[:,'apparent'].to_numpy()))
df_toaster_off_5 = remove_outliers(diff(pd.read_csv("training_data/toaster/toaster_off_5.csv").loc[:,'apparent'].to_numpy()))

# vacuum on
df_vacuum_on_1 = remove_outliers(diff(pd.read_csv("training_data/vacuum/vacuum_on_1.csv").loc[:,'apparent'].to_numpy()))
df_vacuum_on_2 = remove_outliers(diff(pd.read_csv("training_data/vacuum/vacuum_on_2.csv").loc[:,'apparent'].to_numpy()))
df_vacuum_on_3 = remove_outliers(diff(pd.read_csv("training_data/vacuum/vacuum_on_3.csv").loc[:,'apparent'].to_numpy()))
df_vacuum_on_4 = remove_outliers(diff(pd.read_csv("training_data/vacuum/vacuum_on_4.csv").loc[:,'apparent'].to_numpy()))
df_vacuum_on_5 = remove_outliers(diff(pd.read_csv("training_data/vacuum/vacuum_on_5.csv").loc[:,'apparent'].to_numpy()))
# vacuum off
df_vacuum_off_1 = remove_outliers(diff(pd.read_csv("training_data/vacuum/vacuum_off_1.csv").loc[:,'apparent'].to_numpy()))
df_vacuum_off_2 = remove_outliers(diff(pd.read_csv("training_data/vacuum/vacuum_off_2.csv").loc[:,'apparent'].to_numpy()))
df_vacuum_off_3 = remove_outliers(diff(pd.read_csv("training_data/vacuum/vacuum_off_3.csv").loc[:,'apparent'].to_numpy()))
df_vacuum_off_4 = remove_outliers(diff(pd.read_csv("training_data/vacuum/vacuum_off_4.csv").loc[:,'apparent'].to_numpy()))
df_vacuum_off_5 = remove_outliers(diff(pd.read_csv("training_data/vacuum/vacuum_off_5.csv").loc[:,'apparent'].to_numpy()))

# water_kettle on
df_water_kettle_on_1 = remove_outliers(diff(pd.read_csv("training_data/water_kettle/kettle_on_1.csv").loc[:,'apparent'].to_numpy()))
df_water_kettle_on_2 = remove_outliers(diff(pd.read_csv("training_data/water_kettle/kettle_on_2.csv").loc[:,'apparent'].to_numpy()))
df_water_kettle_on_3 = remove_outliers(diff(pd.read_csv("training_data/water_kettle/kettle_on_3.csv").loc[:,'apparent'].to_numpy()))
df_water_kettle_on_4 = remove_outliers(diff(pd.read_csv("training_data/water_kettle/kettle_on_4.csv").loc[:,'apparent'].to_numpy()))
df_water_kettle_on_5 = remove_outliers(diff(pd.read_csv("training_data/water_kettle/kettle_on_5.csv").loc[:,'apparent'].to_numpy()))
# water_kettle off
df_water_kettle_off_1 = remove_outliers(diff(pd.read_csv("training_data/water_kettle/kettle_off_1.csv").loc[:,'apparent'].to_numpy()))
df_water_kettle_off_2 = remove_outliers(diff(pd.read_csv("training_data/water_kettle/kettle_off_2.csv").loc[:,'apparent'].to_numpy()))
df_water_kettle_off_3 = remove_outliers(diff(pd.read_csv("training_data/water_kettle/kettle_off_3.csv").loc[:,'apparent'].to_numpy()))
df_water_kettle_off_4 = remove_outliers(diff(pd.read_csv("training_data/water_kettle/kettle_off_4.csv").loc[:,'apparent'].to_numpy()))
df_water_kettle_off_5 = remove_outliers(diff(pd.read_csv("training_data/water_kettle/kettle_off_5.csv").loc[:,'apparent'].to_numpy()))

training_data = np.array([df_fan_on_1, df_fan_on_2, df_fan_on_3, df_fan_on_4, df_fan_on_5,
                 df_fan_off_1, df_fan_off_2, df_fan_off_3, df_fan_off_4, df_fan_off_5,
                 #df_charger_on_1, df_charger_on_2, df_charger_on_3, df_charger_on_4, df_charger_on_5,
                 #df_charger_off_1, df_charger_off_2, df_charger_off_3, df_charger_off_4, df_charger_off_5,
                 df_hairdryer_on_1, df_hairdryer_on_2, df_hairdryer_on_3, df_hairdryer_on_4, df_hairdryer_on_5,
                 df_hairdryer_off_1, df_hairdryer_off_2, df_hairdryer_off_3, df_hairdryer_off_4, df_hairdryer_off_5,
                 df_lamp_on_1, df_lamp_on_2, df_lamp_on_3, df_lamp_on_4, df_lamp_on_5,
                 df_lamp_off_1, df_lamp_off_2, df_lamp_off_3, df_lamp_off_4, df_lamp_off_5,
                 #df_toaster_on_1, df_toaster_on_2, df_toaster_on_3, df_toaster_on_4, df_toaster_on_5,
                 #df_toaster_off_1, df_toaster_off_2, df_toaster_off_3, df_toaster_off_4, df_toaster_off_5,
                 df_vacuum_on_1, df_vacuum_on_2, df_vacuum_on_3, df_vacuum_on_4, df_vacuum_on_5,
                 df_vacuum_off_1, df_vacuum_off_2, df_vacuum_off_3, df_vacuum_off_4, df_vacuum_off_5,
                 df_water_kettle_on_1, df_water_kettle_on_2, df_water_kettle_on_3, df_water_kettle_on_4, df_water_kettle_on_5,
                 df_water_kettle_off_1, df_water_kettle_off_2, df_water_kettle_off_3, df_water_kettle_off_4, df_water_kettle_off_5])

training_label = np.array([1,1,1,1,1,
                           2,2,2,2,2,
                           #3,3,3,3,3,
                           #4,4,4,4,4,
                           5,5,5,5,5,
                           6,6,6,6,6,
                           7,7,7,7,7,
                           8,8,8,8,8,
                           #9,9,9,9,9,
                           #10,10,10,10,10,
                           11,11,11,11,11,
                           12,12,12,12,12,
                           13,13,13,13,13,
                           14,14,14,14,14])

# 1 - Fan on
# 2 - Fan off
# 3 - Charger on
# 4 - Charger off
# 5 - Hairdryer on
# 6 - Hairdryer off
# 7 - Lamp on
# 8 - Lamp off
# 9 - Toaster on
#10 - Toaster off
#11 - Vacuum on
#12 - Vacuum off
#13 - Water kettle on
#14 - Water kettle of

model = create_model()
train_model(training_data, training_label, model)

model.save("test_model_2")

# End of training process
'''

# load the whole model with weights in this file and start metering with load disaggregation (10 Hz mode)
model = tf.keras.models.load_model("test_model_2")
metering_with_ld("test_with_ld_and_states.csv")

# load the whole model with weights in this file and predict for the given file
#model = tf.keras.models.load_model("test_model_1")
#predict("smartmeter_data/data.csv")

'''
x = range(0,len(df_fan_on_1))
plt.figure(1)
plt.plot(x,df_fan_on_1, 'r')
plt.plot(x,df_fan_on_2, 'r')
plt.plot(x,df_fan_on_3, 'r')
plt.plot(x,df_fan_on_4, 'r')
plt.plot(x,df_fan_on_5, 'r')
plt.plot(x,df_fan_off_1, 'b')
plt.plot(x,df_fan_off_2, 'b')
plt.plot(x,df_fan_off_3, 'b')
plt.plot(x,df_fan_off_4, 'b')
plt.plot(x,df_fan_off_5, 'b')
plt.title('fan')
plt.figure(2)
plt.plot(x,df_charger_on_1, 'r')
plt.plot(x,df_charger_on_2, 'r')
plt.plot(x,df_charger_on_3, 'r')
plt.plot(x,df_charger_on_4, 'r')
plt.plot(x,df_charger_on_5, 'r')
plt.plot(x,df_charger_off_1, 'b')
plt.plot(x,df_charger_off_2, 'b')
plt.plot(x,df_charger_off_3, 'b')
plt.plot(x,df_charger_off_4, 'b')
plt.plot(x,df_charger_off_5, 'b')
plt.title('charger')
plt.figure(3)
plt.plot(x,df_hairdryer_on_1, 'r')
plt.plot(x,df_hairdryer_on_2, 'r')
plt.plot(x,df_hairdryer_on_3, 'r')
plt.plot(x,df_hairdryer_on_4, 'r')
plt.plot(x,df_hairdryer_on_5, 'r')
plt.plot(x,df_hairdryer_off_1, 'b')
plt.plot(x,df_hairdryer_off_2, 'b')
plt.plot(x,df_hairdryer_off_3, 'b')
plt.plot(x,df_hairdryer_off_4, 'b')
plt.plot(x,df_hairdryer_off_5, 'b')
plt.title('hairdryer')
plt.figure(4)
plt.plot(x,df_lamp_on_1, 'r')
plt.plot(x,df_lamp_on_2, 'r')
plt.plot(x,df_lamp_on_3, 'r')
plt.plot(x,df_lamp_on_4, 'r')
plt.plot(x,df_lamp_on_5, 'r')
plt.plot(x,df_lamp_off_1, 'b')
plt.plot(x,df_lamp_off_2, 'b')
plt.plot(x,df_lamp_off_3, 'b')
plt.plot(x,df_lamp_off_4, 'b')
plt.plot(x,df_lamp_off_5, 'b')
plt.title('lamp')
plt.figure(5)
plt.plot(x,df_toaster_on_1, 'r')
plt.plot(x,df_toaster_on_2, 'r')
plt.plot(x,df_toaster_on_3, 'r')
plt.plot(x,df_toaster_on_4, 'r')
plt.plot(x,df_toaster_on_5, 'r')
plt.plot(x,df_toaster_off_1, 'b')
plt.plot(x,df_toaster_off_2, 'b')
plt.plot(x,df_toaster_off_3, 'b')
plt.plot(x,df_toaster_off_4, 'b')
plt.plot(x,df_toaster_off_5, 'b')
plt.title('toaster')
plt.figure(6)
plt.plot(x,df_vacuum_on_1, 'r')
plt.plot(x,df_vacuum_on_2, 'r')
plt.plot(x,df_vacuum_on_3, 'r')
plt.plot(x,df_vacuum_on_4, 'r')
plt.plot(x,df_vacuum_on_5, 'r')
plt.plot(x,df_vacuum_off_1, 'b')
plt.plot(x,df_vacuum_off_2, 'b')
plt.plot(x,df_vacuum_off_3, 'b')
plt.plot(x,df_vacuum_off_4, 'b')
plt.plot(x,df_vacuum_off_5, 'b')
plt.title('vacuum')
plt.figure(7)
plt.plot(x,df_water_kettle_on_1, 'r')
plt.plot(x,df_water_kettle_on_2, 'r')
plt.plot(x,df_water_kettle_on_3, 'r')
plt.plot(x,df_water_kettle_on_4, 'r')
plt.plot(x,df_water_kettle_on_5, 'r')
plt.plot(x,df_water_kettle_off_1, 'b')
plt.plot(x,df_water_kettle_off_2, 'b')
plt.plot(x,df_water_kettle_off_3, 'b')
plt.plot(x,df_water_kettle_off_4, 'b')
plt.plot(x,df_water_kettle_off_5, 'b')
plt.title('water kettle')
plt.show()
'''