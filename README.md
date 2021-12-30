# Welcome!

This work is an extension of the YoMoPie Python library (https://github.com/klemenjak/YoMoPie) and extends the functionality of the code with a Non-Intrusive Load Monitoring (NILM) algorithm. This allows to read out individual consumers from the measured power consumption. 
Since this paper does not go into the functionalities of YoMoPies in detail, it is recommended to first study the work linked above.
## Installation

Since the YoMoPie library itself has also been slightly modified, it is recommended to use the YoMoPie Python file from this repository.
## Examples of use
The file smart_metering.py contains all functions of the NILM algorithm and the file can be extended and used as desired. If the functionality is not changed, the individual functions should not be changed. The use of the file is as follows. 
First, a YoMoPie object must be created:
```python
yomo = yomopie.YoMoPie()
```
The two arrays device_states and timings are bound to the number of devices and must be changed if necessary (the comments next to the arrays describe which devices were used).
```python
device_states = [0,0,0,0,0,0,0]
timings = [0,0,0,0,0,0,0]
```

Before the system can be started, a model must first be trained (line 195 down) or an existing model must be loaded (test_model_2 was trained on the devices used in the tests of the algorithm).
```python
model = load_model("test_model_2")
```
Now you can either start the system just like that and save the data in the folder 'data' with the start time.
```python
start_metering()
```
If the data is to be saved with a special name, this function is used.
```python
metering_with_ld("test_logfile")
```
Last but not least, an already existing CSV file can be disaggregated.
```python
predict_file("file_prediction_test")
```

# Code documentation
**[Imports](#imports)**<br>
**[Methods](#methods)**<br>
*[-create_model](#create_model)*<br>
*[-train_model](#train_model)*<br>
*[-save_model](#save_model)*<br>
*[-load_model](#load_model)*<br>
*[-create_database](#create_database)*<br>
*[-diff](#diff)*<br>
*[-change_state](#change_state)*<br>
*[-write_to_logfile](#write_to_logfile)*<br>
*[-predict_file](#predict_file)*<br>
*[-metering_with_ld](#metering_with_ld)*<br>
*[-start_metering](#start_metering)*<br>


## Imports
The algorithm requires some additional libraries:

**YoMoPie**: The YoMoPie package is used for measuring the energy consumption.

**tensorflow**: The time package is required to create the keras machine learning model.

**pandas**: Pandas is required to read and write to the CSV files.

**numpy**: The numpy package is used to do array operations for feeding the model.

**time**: time is used for creating time stamps.

**math**: The math package is used for mathematical operations.

**datetime**: This package allowes to convert time stamps to a readable date format.

```python
import YoMoPie as yomopie
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import math
import datetime
```

## Methods

In this Section, we describe every method of the python file. A description of function parameters and return values is given.

### create_model

**Description**: This method creates a sequential keras model and compiles it.

**Parameters**:

* number_devices - The number of devices used in the system

**Returns**: The model.

```python
def create_model(number_devices):
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(1),
                                        tf.keras.layers.Dense(64),
                                        tf.keras.layers.Dense(number_devices, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Model compiled successful")
    return model
```

### train_model

**Description**: Trains the model. The training data should be one big array like [[Device 1 data, Device 2 data, ..., Device n data]]. Same for the trainig labels. 

**Parameters**: 

* training_data - The data thats used to train the model (in array form)

* training_labels - The labels for the data (same array dimension as the training_data)

* model - The model that will be trained

**Returns**: Nothing.

```python
def train_model(training_data, training_labels, model):
    model.fit(training_data, training_labels, epochs=10000)
    return 0
```

### save_model

**Description**: This function saves the model to a file with all weights and properties.

**Parameters**:

* filename - How the file will be named where the model is saved

**Returns**: Nothing.

```python
def save_model(filename):
    model.save(filename)
    return 0
```

### load_model

**Description**: Loads a model into the system so there is no need to train the model each time the programm is started.

**Parameters**: 

* filename - The name of the file that will be loaded as model

**Returns**: The model.

```python
def load_model(filename):
    return tf.keras.models.load_model(filename)
```

### create_database

**Description**: Creates a CSV file with 50 samples and saves it to the file name given.

**Parameters**: 
* filename - The name of the CSV file where the 50 samples will be saved

**Returns**: Nothing.

```python
def create_database(filename):
    yomo.disable_board()
    time.sleep(1)
    yomo.enable_board()
    time.sleep(1)    
    yomo.create_dataset(50,0.1, filename)
    return 0
```

### diff

**Description**: Calculates the absolute difference between the last and first value of the array.

**Parameters**: 
* array - the array from which the difference shall be calculated

**Returns**: The absolute difference.

```python
def diff(array):
    #print("Diff: %f -  %f = %f" %(array[len(array)-1], array[0], array[len(array)-1]-array[0]))
    return array[len(array)-1]-array[0]
```		
### change_state    

**Description**: Changes the status of the devices by converting the values 1 to 14 to 0 to 6 and, depending on the action, changes the value in the device_states array and calls the write_to_logfile function.

**Parameters**:

* state - The state change that the model predicted

**Returns**: Nothing.

```python
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
```

### write_to_logfile

**Description**: Writes an entry in the log file. A text is created with the unit number and the action (0 for switch off and 1 for switch on) which is then written into the log file.

**Parameters**:
* device - The device that the model predicted

* state - The state change that the model predicted (on of off)

**Returns**: Nothing.

```python
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
```

### predict_file

**Description**: Takes the transferred file and creates a new CSV file for it in which the disaggregated device statuses are listed.

**Parameters**:

* filename - The name of the file that will be disaggregated

**Returns**: Nothing.

```python
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
```		

### metering_with_ld  

**Description**: Starts endless measurement of power consumption and saves it to a CSV file. In addition, the disaggregated device statuses are saved in an extra file and, if necessary, also in the log file.

**Parameters**:
* name - The filename of this measurment 

**Returns**: Nothing.

```python
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
```

### start_metering        

**Description**: Starts endless measurement of power consumption and saves it to a CSV file (unlike the metering_with_ld function, no file name is given here and the data is saved in the folder 'data' with the start time as the file name. In addition, the disaggregated device statuses are saved in an extra file and, if necessary, also in the log file. 

**Parameters**: None.

**Returns**: Nothing.

```python
def start_metering():
    now = datetime.datetime.now()
    filename = "data/" + now.strftime("%Y") + "_"  + now.strftime("%m") + "_" + now.strftime("%d") + "_" + now.strftime("%H_%M_%S")
    print("#### Data will be stored in " + filename + " ####")
    metering_with_ld(filename)
    return 0
```
