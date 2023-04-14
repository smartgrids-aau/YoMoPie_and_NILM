# Smart meter data acquisition and disaggregation module
#
# Code by Stefan Jost
# License: GPL v3

import YoMoPie as yomopie
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import math
import datetime


yomo = yomopie.YoMoPie()

# 1 - Fan
# 2 - Charger (not implemented, because this device does have more then only 2 states)
# 3 - Hair dryer
# 4 - Lamp
# 5 - Toaster (not implemented, because its power consumption is same like the power consumption of the hair dryer)
# 6 - Vacuum
# 7 - Water kettle
device_states = [0,0,0,0,0,0,0]
timings = [0,0,0,0,0,0,0]

def metering_with_ld(name):
    file = name + ".csv"
    if file =='':
        print('Unsupported filename!')
        return 1
    device_states_filename = name + "_device_states.csv"
    logfile = open(device_states_filename,'a')
    logfile.write("time;Fan;Charger;Hair dryer;Lamp;Toaster;Vacuum;Water kettle;\n")
    logfile = open(file,'a')
    logfile.write("time;apparent;\n")
    yomo.sample_intervall = 0.1
    yomo.disable_board()
    time.sleep(1)
    yomo.enable_board()
    time.sleep(1)
    step = 0
    cp = [0,0]
    samples = []
    while(1):
        step = step + 1
        sample = []
        sample.append(time.time())
        power=yomo.get_apparent_energy()[1]
        sample.append(power)
        samples.append(power)
        if step >= 10 and len(samples)>50:
            prediction = model.predict(np.array([diff(samples[len(samples)-50:len(samples)])]))
            print("----------------%f|%f----------------" % (np.argmax(prediction), np.amax(prediction)))
            if np.amax(prediction)>0.8: # confidence should be in the range of 70% to 90% to get the best accuracy
                cp.append(np.argmax(prediction))
                if cp[len(cp)-1] == np.argmax(prediction) and cp[len(cp)-2] == np.argmax(prediction):
                    change_state(np.argmax(prediction))
            step = 0
            logfile = open(device_states_filename,'a')
            logfile.write("%s;" % time.time())
            for value in device_states:
                logfile.write("%s;" % value)
            logfile.write("\n")
            yomo.get_apparent_energy() #to reset the register and avoid the 10ms added power from the routine time
        logfile = open(file,'a')
        for value in sample:
            logfile.write("%s;" % value)
        logfile.write("\n")
        time.sleep(0.1);
    return 0

def write_to_logfile(device, state):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y") + "/"  + now.strftime("%m") + "/" + now.strftime("%d") + " " + now.strftime("%H:%M:%S")
    switcher_device = {
        0: "Fan",
        1: "Charger",
        2: "Hair dryer",
        3: "Lamp",
        4: "Toaster",
        5: "Vacuum",
        6: "Water kettle"}
    switcher_action = {
        0: " turned off",
        1: " turned on"}
    
    # Power consumption calculated from database
    switcher_power = {
        0: 31.77,
        1: 0,
        2: 1816.84,
        3: 153.13,
        4: 0,
        5: 642.69,
        6: 797.77}
    if state == 0:
        delta = now-timings[device]
        text = timestamp + ": " + switcher_device.get(device, "Invalid device") + switcher_action.get(state, "Invalid action") + " (" +  str(round(delta.total_seconds()/3600 * switcher_power.get(device, 0),2)) +  " Wh in " + str(datetime.timedelta(days=delta.days, seconds=delta.seconds)) + ")\n"
    else:
        text = timestamp + ": " + switcher_device.get(device, "Invalid device") + switcher_action.get(state, "Invalid action") + "\n"
    logfile = open("logfile.txt",'a')    
    logfile.write(text)
    print(text)
    return 0

def start_metering():
    now = datetime.datetime.now()
    filename = "data/" + now.strftime("%Y") + "_"  + now.strftime("%m") + "_" + now.strftime("%d") + "_" + now.strftime("%H_%M_%S")
    print("#### Data will be stored in " + filename + " ####")
    metering_with_ld(filename)
    return 0

def change_state(state):
    device = math.ceil(state/2)-1
    temp_states = device_states.copy()
    if state%2==0:
        device_states[device] = 0
        if not np.array_equiv(device_states,temp_states):            
            temp_states = device_states.copy()
            write_to_logfile(device, 0)
            timings[device] = datetime.datetime.now()
    else:
        device_states[device] = 1
        if not np.array_equiv(device_states,temp_states):
            temp_states = device_states.copy() 
            timings[device] = datetime.datetime.now()
            write_to_logfile(device, 1)
    return 0

def diff(array):
    #print("Diff: %f -  %f = %f" %(array[len(array)-1], array[0], array[len(array)-1]-array[0]))
    return array[len(array)-1]-array[0]

def create_model(number_devices):
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(1),
                                        tf.keras.layers.Dense(64),
                                        tf.keras.layers.Dense(number_devices, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Model compiled successful")
    return model

def save_model(filename):
    model.save(filename)
    return 0

def load_model(filename):
    return tf.keras.models.load_model(filename)

def train_model(training_data, training_labels, model):
    model.fit(training_data, training_labels, epochs=10000)
    return 0

def predict_file(filename):
    df = pd.read_csv(filename + ".csv", sep = ';').loc[:,'apparent'].to_numpy()
    device_states_filename = filename + "_device_states.csv"
    logfile = open(device_states_filename,'a')
	# names of the devices used
    logfile.write("time; Fan; Charger; Hair dryer; Lamp; Toaster; Vacuum; Water kettle; \n")
    cp = [0,0]
    for i in range(0, len(df)-49, 10):
        prediction = model.predict(np.array([diff(df[i:i+49])]))
        print("----------------%f|%f----------------" % (np.argmax(prediction), np.amax(prediction)))
        if np.amax(prediction)>0.8: # confidence should be in the range of 70% to 90% to get the best accuracy
            cp.append(np.argmax(prediction))
            if cp[len(cp)-1] == np.argmax(prediction) and cp[len(cp)-2] == np.argmax(prediction):
                change_state(np.argmax(prediction))
        logfile = open(device_states_filename,'a')
        logfile.write("%s; " % time.time())
        for value in device_states:
            logfile.write("%s; " % value)
        logfile.write("\n")
        print(device_states)
    return 0
    
# create a dataset with 50 samples and 10 hz
def create_database(filename):
    yomo.disable_board()
    time.sleep(1)
    yomo.enable_board()
    time.sleep(1)    
    yomo.create_dataset(50,0.1, filename)
    return 0
        
    
# load the whole model with weights in this file
model = load_model("test_model_2")

# start metering without filename. The data will be stored in a file named by the starting date and time in the folder 'data'.
#start_metering()

# start metering and save the data in the file given
metering_with_ld("test_logfile")

# predict for the given file (without .csv)
#predict_file("file_prediction_test")

'''
# Start of training process
# 1. read training data
# 2. create model
# 3. train model
# 4. save model local

# fan on
df_fan_on_1 = diff(pd.read_csv("training_data/fan/fan_on_1.csv").loc[:,'apparent'].to_numpy())
df_fan_on_2 = diff(pd.read_csv("training_data/fan/fan_on_2.csv").loc[:,'apparent'].to_numpy())
df_fan_on_3 = diff(pd.read_csv("training_data/fan/fan_on_3.csv").loc[:,'apparent'].to_numpy())
df_fan_on_4 = diff(pd.read_csv("training_data/fan/fan_on_4.csv").loc[:,'apparent'].to_numpy())
df_fan_on_5 = diff(pd.read_csv("training_data/fan/fan_on_5.csv").loc[:,'apparent'].to_numpy())
# fan off
df_fan_off_1 = diff(pd.read_csv("training_data/fan/fan_off_1.csv").loc[:,'apparent'].to_numpy())
df_fan_off_2 = diff(pd.read_csv("training_data/fan/fan_off_2.csv").loc[:,'apparent'].to_numpy())
df_fan_off_3 = diff(pd.read_csv("training_data/fan/fan_off_3.csv").loc[:,'apparent'].to_numpy())
df_fan_off_4 = diff(pd.read_csv("training_data/fan/fan_off_4.csv").loc[:,'apparent'].to_numpy())
df_fan_off_5 = diff(pd.read_csv("training_data/fan/fan_off_5.csv").loc[:,'apparent'].to_numpy())

# charger on
df_charger_on_1 = diff(pd.read_csv("training_data/charger/charger_on_1.csv").loc[:,'apparent'].to_numpy())
df_charger_on_2 = diff(pd.read_csv("training_data/charger/charger_on_2.csv").loc[:,'apparent'].to_numpy())
df_charger_on_3 = diff(pd.read_csv("training_data/charger/charger_on_3.csv").loc[:,'apparent'].to_numpy())
df_charger_on_4 = diff(pd.read_csv("training_data/charger/charger_on_4.csv").loc[:,'apparent'].to_numpy())
df_charger_on_5 = diff(pd.read_csv("training_data/charger/charger_on_5.csv").loc[:,'apparent'].to_numpy())
# charger off
df_charger_off_1 = diff(pd.read_csv("training_data/charger/charger_off_1.csv").loc[:,'apparent'].to_numpy())
df_charger_off_2 = diff(pd.read_csv("training_data/charger/charger_off_2.csv").loc[:,'apparent'].to_numpy())
df_charger_off_3 = diff(pd.read_csv("training_data/charger/charger_off_3.csv").loc[:,'apparent'].to_numpy())
df_charger_off_4 = diff(pd.read_csv("training_data/charger/charger_off_4.csv").loc[:,'apparent'].to_numpy())
df_charger_off_5 = diff(pd.read_csv("training_data/charger/charger_off_5.csv").loc[:,'apparent'].to_numpy())

# hairdryer on
df_hairdryer_on_1 = diff(pd.read_csv("training_data/hairdryer/hairdryer_on_1.csv").loc[:,'apparent'].to_numpy())
df_hairdryer_on_2 = diff(pd.read_csv("training_data/hairdryer/hairdryer_on_2.csv").loc[:,'apparent'].to_numpy())
df_hairdryer_on_3 = diff(pd.read_csv("training_data/hairdryer/hairdryer_on_3.csv").loc[:,'apparent'].to_numpy())
df_hairdryer_on_4 = diff(pd.read_csv("training_data/hairdryer/hairdryer_on_4.csv").loc[:,'apparent'].to_numpy())
df_hairdryer_on_5 = diff(pd.read_csv("training_data/hairdryer/hairdryer_on_5.csv").loc[:,'apparent'].to_numpy())
# hairdryer off
df_hairdryer_off_1 = diff(pd.read_csv("training_data/hairdryer/hairdryer_off_1.csv").loc[:,'apparent'].to_numpy())
df_hairdryer_off_2 = diff(pd.read_csv("training_data/hairdryer/hairdryer_off_2.csv").loc[:,'apparent'].to_numpy())
df_hairdryer_off_3 = diff(pd.read_csv("training_data/hairdryer/hairdryer_off_3.csv").loc[:,'apparent'].to_numpy())
df_hairdryer_off_4 = diff(pd.read_csv("training_data/hairdryer/hairdryer_off_4.csv").loc[:,'apparent'].to_numpy())
df_hairdryer_off_5 = diff(pd.read_csv("training_data/hairdryer/hairdryer_off_5.csv").loc[:,'apparent'].to_numpy())

# lamp on
df_lamp_on_1 = diff(pd.read_csv("training_data/lamp/lamp_on_1.csv").loc[:,'apparent'].to_numpy())
df_lamp_on_2 = diff(pd.read_csv("training_data/lamp/lamp_on_2.csv").loc[:,'apparent'].to_numpy())
df_lamp_on_3 = diff(pd.read_csv("training_data/lamp/lamp_on_3.csv").loc[:,'apparent'].to_numpy())
df_lamp_on_4 = diff(pd.read_csv("training_data/lamp/lamp_on_4.csv").loc[:,'apparent'].to_numpy())
df_lamp_on_5 = diff(pd.read_csv("training_data/lamp/lamp_on_5.csv").loc[:,'apparent'].to_numpy())
# lamp off
df_lamp_off_1 = diff(pd.read_csv("training_data/lamp/lamp_off_1.csv").loc[:,'apparent'].to_numpy())
df_lamp_off_2 = diff(pd.read_csv("training_data/lamp/lamp_off_2.csv").loc[:,'apparent'].to_numpy())
df_lamp_off_3 = diff(pd.read_csv("training_data/lamp/lamp_off_3.csv").loc[:,'apparent'].to_numpy())
df_lamp_off_4 = diff(pd.read_csv("training_data/lamp/lamp_off_4.csv").loc[:,'apparent'].to_numpy())
df_lamp_off_5 = diff(pd.read_csv("training_data/lamp/lamp_off_5.csv").loc[:,'apparent'].to_numpy())

# toaster on
df_toaster_on_1 = diff(pd.read_csv("training_data/toaster/toaster_on_1.csv").loc[:,'apparent'].to_numpy())
df_toaster_on_2 = diff(pd.read_csv("training_data/toaster/toaster_on_2.csv").loc[:,'apparent'].to_numpy())
df_toaster_on_3 = diff(pd.read_csv("training_data/toaster/toaster_on_3.csv").loc[:,'apparent'].to_numpy())
df_toaster_on_4 = diff(pd.read_csv("training_data/toaster/toaster_on_4.csv").loc[:,'apparent'].to_numpy())
df_toaster_on_5 = diff(pd.read_csv("training_data/toaster/toaster_on_5.csv").loc[:,'apparent'].to_numpy())
# toaster off
df_toaster_off_1 = diff(pd.read_csv("training_data/toaster/toaster_off_1.csv").loc[:,'apparent'].to_numpy())
df_toaster_off_2 = diff(pd.read_csv("training_data/toaster/toaster_off_2.csv").loc[:,'apparent'].to_numpy())
df_toaster_off_3 = diff(pd.read_csv("training_data/toaster/toaster_off_3.csv").loc[:,'apparent'].to_numpy())
df_toaster_off_4 = diff(pd.read_csv("training_data/toaster/toaster_off_4.csv").loc[:,'apparent'].to_numpy())
df_toaster_off_5 = diff(pd.read_csv("training_data/toaster/toaster_off_5.csv").loc[:,'apparent'].to_numpy())

# vacuum on
df_vacuum_on_1 = diff(pd.read_csv("training_data/vacuum/vacuum_on_1.csv").loc[:,'apparent'].to_numpy())
df_vacuum_on_2 = diff(pd.read_csv("training_data/vacuum/vacuum_on_2.csv").loc[:,'apparent'].to_numpy())
df_vacuum_on_3 = diff(pd.read_csv("training_data/vacuum/vacuum_on_3.csv").loc[:,'apparent'].to_numpy())
df_vacuum_on_4 = diff(pd.read_csv("training_data/vacuum/vacuum_on_4.csv").loc[:,'apparent'].to_numpy())
df_vacuum_on_5 = diff(pd.read_csv("training_data/vacuum/vacuum_on_5.csv").loc[:,'apparent'].to_numpy())
# vacuum off
df_vacuum_off_1 = diff(pd.read_csv("training_data/vacuum/vacuum_off_1.csv").loc[:,'apparent'].to_numpy())
df_vacuum_off_2 = diff(pd.read_csv("training_data/vacuum/vacuum_off_2.csv").loc[:,'apparent'].to_numpy())
df_vacuum_off_3 = diff(pd.read_csv("training_data/vacuum/vacuum_off_3.csv").loc[:,'apparent'].to_numpy())
df_vacuum_off_4 = diff(pd.read_csv("training_data/vacuum/vacuum_off_4.csv").loc[:,'apparent'].to_numpy())
df_vacuum_off_5 = diff(pd.read_csv("training_data/vacuum/vacuum_off_5.csv").loc[:,'apparent'].to_numpy())

# water_kettle on
df_water_kettle_on_1 = diff(pd.read_csv("training_data/water_kettle/kettle_on_1.csv").loc[:,'apparent'].to_numpy())
df_water_kettle_on_2 = diff(pd.read_csv("training_data/water_kettle/kettle_on_2.csv").loc[:,'apparent'].to_numpy())
df_water_kettle_on_3 = diff(pd.read_csv("training_data/water_kettle/kettle_on_3.csv").loc[:,'apparent'].to_numpy())
df_water_kettle_on_4 = diff(pd.read_csv("training_data/water_kettle/kettle_on_4.csv").loc[:,'apparent'].to_numpy())
df_water_kettle_on_5 = diff(pd.read_csv("training_data/water_kettle/kettle_on_5.csv").loc[:,'apparent'].to_numpy())
# water_kettle off
df_water_kettle_off_1 = diff(pd.read_csv("training_data/water_kettle/kettle_off_1.csv").loc[:,'apparent'].to_numpy())
df_water_kettle_off_2 = diff(pd.read_csv("training_data/water_kettle/kettle_off_2.csv").loc[:,'apparent'].to_numpy())
df_water_kettle_off_3 = diff(pd.read_csv("training_data/water_kettle/kettle_off_3.csv").loc[:,'apparent'].to_numpy())
df_water_kettle_off_4 = diff(pd.read_csv("training_data/water_kettle/kettle_off_4.csv").loc[:,'apparent'].to_numpy())
df_water_kettle_off_5 = diff(pd.read_csv("training_data/water_kettle/kettle_off_5.csv").loc[:,'apparent'].to_numpy())

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

model = create_model(15)
train_model(training_data, training_label, model)
save_model("test_model_2")

# End of training process
'''