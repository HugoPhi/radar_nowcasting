import os
import numpy as np

def load_data(main_dir, altitude):
    main_dir = main_dir  # root directory of the dataset
    target_altitude = altitude  # target altitude
    
    num = 0
    temp = {}  # a dict to store the data of (frame_number, frames, 256, 256), key -> feature
    datasets = {}  # return value, a dict of datasets shpes (features, time_steps, 256, 256), key -> frame_number
    
    for variable in ['dBZ', 'ZDR', 'KDP']:  # traverse each feature 
        temp[variable] = []
        variable_dir = os.path.join(main_dir, variable, target_altitude)
        
        num = len(os.listdir(variable_dir))  # number of datasets
        for data_dir in os.listdir(variable_dir):
            data_dir_path = os.path.join(variable_dir, data_dir)
            
            frames = []  # read all frames 
            for frame_file in os.listdir(data_dir_path):
                if frame_file.endswith('.npy'):
                    frame_path = os.path.join(data_dir_path, frame_file)
                    frame_data = np.load(frame_path)
                    frames.append(frame_data)
            
            dataset = np.stack(frames, axis=0)  # (time_steps, 256, 256)
            
            temp[variable].append(dataset)  # (frame_number + 1, frames', 256, 256)
    
    for x in range(num):
        # datasets[x] = np.array([temp[y][num - x - 1] for y in ('dBZ', 'ZDR', 'KDP')])
        datasets[x] = [temp[y][num - x - 1] for y in ('dBZ', 'ZDR', 'KDP')]
    
    return datasets  # (frame_number, features, frames, 256, 256), features: dBZ -> 0, ZDR -> 1, KDP -> 2 

if __name__ == '__main__':
    dir = '/root/CodeHub/py/radar-pol-wforcast/.data/new_2308_1'

    datasets = load_data(dir, '1.0km')
    for x in range(len(datasets)):
        print(f'dataset {x} --> {np.array(datasets[x]).shape}')
else:
    pass

