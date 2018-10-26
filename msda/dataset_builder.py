import numpy as np
import pandas as pd

def get_ds_infos():
    """
    Read the file includes data subject information.
    
    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]
    
    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes 
    """ 

    dss = pd.read_csv("data_subjects_info.csv")
    print("[INFO] -- Data subjects' information is imported.")
    
    return dss

def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.
    
    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t+".x",t+".y",t+".z"])
        else:
            dt_list.append([t+".roll", t+".pitch", t+".yaw"])

    return dt_list


def creat_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=True, combine_grav_acc=False):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.
        combine_grav_acc: True, means adding each axis of gravity to  corresponding axis of userAcceleration.
    Returns: 
        It returns a time-series of sensor data.
    
    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    if labeled:
        dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 
    else:
        dataset = np.zeros((0,num_data_cols))
        
    ds_list = get_ds_infos()
    
    print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list["code"]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = 'A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))
                
                if combine_grav_acc:
                    raw_data["userAcceleration.x"] = raw_data["userAcceleration.x"].add(raw_data["gravity.x"])
                    raw_data["userAcceleration.y"] = raw_data["userAcceleration.y"].add(raw_data["gravity.y"])
                    raw_data["userAcceleration.z"] = raw_data["userAcceleration.z"].add(raw_data["gravity.z"])
                
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        
                    else:
                        vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                    vals = vals[:,:num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                            sub_id-1,
                            ds_list["weight"][sub_id-1],
                            ds_list["height"][sub_id-1],
                            ds_list["age"][sub_id-1],
                            ds_list["gender"][sub_id-1],
                            trial          
                           ]]*len(raw_data))
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset,vals, axis=0)
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]
            
    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]
    
    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset
#________________________________
#________________________________

def ts_to_secs(dataset, w, s, standardize = False, **options):
    
    data = dataset[dataset.columns[:-7]].values    
    act_labels = dataset["act"].values
    id_labels = dataset["id"].values
    trial_labels = dataset["trial"].values

    mean = 0
    std = 1
    if standardize:
        ## Standardize each sensor’s data to have a zero mean and unity standard deviation.
        ## As usual, we normalize test dataset by training dataset's parameters 
        if options:
            mean = options.get("mean")
            std = options.get("std")
            print("[INFO] -- Test Data has been standardized")
        else:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            print("[INFO] -- Training Data has been standardized: the mean is = "+str(mean)+" ; and the std is = "+str(std))            

        data -= mean
        data /= std
    else:
        print("[INFO] -- Without Standardization.....")

    ## We want the Rows of matrices show each Feature and the Columns show time points.
    data = data.T

    m = data.shape[0]   # Data Dimension 
    ttp = data.shape[1] # Total Time Points
    number_of_secs = int(round(((ttp - w)/s)))

    ##  Create a 3D matrix for Storing Sections  
    secs_data = np.zeros((number_of_secs , m , w ))
    act_secs_labels = np.zeros(number_of_secs)
    id_secs_labels = np.zeros(number_of_secs)

    k=0
    for i in range(0 , ttp-w, s):
        j = i // s
        if j >= number_of_secs:
            break
        if id_labels[i] != id_labels[i+w-1]: 
            continue
        if act_labels[i] != act_labels[i+w-1]: 
            continue
        if trial_labels[i] != trial_labels[i+w-1]:
            continue
            
        secs_data[k] = data[:, i:i+w]
        act_secs_labels[k] = act_labels[i].astype(int)
        id_secs_labels[k] = id_labels[i].astype(int)
        k = k+1
        
    secs_data = secs_data[0:k]
    act_secs_labels = act_secs_labels[0:k]
    id_secs_labels = id_secs_labels[0:k]
    return secs_data, act_secs_labels, id_secs_labels, mean, std
##________________________________________________________________