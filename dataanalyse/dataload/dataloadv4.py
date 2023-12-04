import os
import numpy as np

def load_data(main_dir, altitude, dirid, frameid):
    main_dir = main_dir  # root directory of the dataset
    
    num = 0
    temp = {}  # a dict to store the data of (frame_number, frames, 256, 256), key -> feature
    datasets = {}  # return value, a dict of datasets shpes (features, time_steps, 256, 256), key -> frame_number
    
    for variable in ['dBZ', 'ZDR', 'KDP']:  # traverse each feature 
        temp[variable] = []
        for target_altitude in altitude:
            variable_dir = os.path.join(main_dir, variable, target_altitude)
            
            frame_path = os.path.join(variable_dir, f'data_dir_{dirid:03d}', f'frame_{frameid:03d}.npy')
            frame_data = np.load(frame_path)
            temp[variable].append(frame_data)
    
    # return temp['KDP'], temp['ZDR'], temp['dBZ']
    return np.array(temp['KDP']), np.array(temp['ZDR']), np.array(temp['dBZ'])

if __name__ == '__main__':
    dir = '/root/CodeHub/py/radar-pol-wforcast/.data/new_2308_1'

    datasets = load_data(dir, ('1.0km', '3.0km', '7.0km'), 1, 1)
    for x in range(len(datasets)):
        # print(f'dataset {x} --> {np.array(datasets[x]).shape}')
        print(f'dataset {x} --> {datasets[x].shape}')
else:
    pass

