import numpy as np
from scipy.linalg import norm
import dataload.dataloadv4 as dl
from pykalman import KalmanFilter

def slide_windows(data, window_size=10, stride=5):
    seqs = []
    lendata = len(data[0])
    # use slide windows
    for i in range(0, lendata - window_size + 1, stride):
        seq = data[:, i:i+window_size, :, :]
        seqs.append(seq)
    return seqs

def split_dataset(X, y, split_ratio=0.8):
    # Split the data into training and validation sets
    split_index = int(split_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test


def kalman_filter_matrix(data):
    batch_size, time_steps, height, width, channels = data.shape
    data_reshaped = data.reshape((batch_size, time_steps, -1, channels))

    kf = KalmanFilter(
        initial_state_mean=np.zeros(data_reshaped.shape[2]),
        n_dim_obs=data_reshaped.shape[2]
    )
    kf = kf.em(data_reshaped[:, :, :, 0], n_iter=10)  # You might need to adjust n_iter
    filtered_state_means, _ = kf.filter(data_reshaped[:, :, :, 0])

    filtered_data = filtered_state_means.reshape((batch_size, time_steps, height, width, channels))

    return filtered_data


def kalman_filter_iterative(data):
    batch_size, time_steps, height, width, channels = data.shape
    filtered_data = np.zeros_like(data)

    for i in range(batch_size):
        for j in range(height):
            for k in range(width):
                for channel in range(channels):
                    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
                    for t in range(time_steps):
                        observation = data[i, t, j, k, channel]
                        if t == 0:
                            kf = kf.em(observation)
                        else:
                            (filtered_state_means, _) = kf.filter_update(filtered_state_means[-1], kf.transition_matrix, observation)
                            kf = kf.filter_update(filtered_state_means[-1], kf.transition_matrix, observation)

                        filtered_data[i, t, j, k, channel] = filtered_state_means.flatten()

    return filtered_data

kalman_filter = kalman_filter_iterative

def load_xy(datasets, sample_dir_size, normalparam, split_ratio=0.8, window_size=10, stride=5):
    feature_num = len(normalparam)
    samples = np.array([]).reshape(0, feature_num, window_size, 256, 256)
    
    for id in range(sample_dir_size):
        samples = np.concatenate((samples, np.array(slide_windows(np.array(datasets[id]), window_size, stride))), axis=0)
    
    for x in range(feature_num):
        samples = samples.swapaxes(x+1, x+2)
    
    # normalization 
    for x in range(feature_num):
        mmin, mmax = normalparam[x]
        samples[:, :, :, :, x] = (samples[:, :, :, :, x] - mmin) / (mmax - mmin)
    
    # make X and y
    X = samples[:-window_size, :, :, :, :]
    y = samples[window_size:, :, :, :, 0:1]
    
    # split train and test
    X_train, X_test, y_train, y_test = split_dataset(X, y, split_ratio)
    
    print(f'X_train.shape: {X_train.shape}')
    print(f'X_test.shape:  {X_test.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'y_test.shape:  {y_test.shape}')
    return X_train, X_test, y_train, y_test



normalparam = ([0, 65], [-1, 5], [-1, 6])  # dBZ: [0, 65], ZDR: [-1, 5], KDP: [-1, 6]

if __name__ == '__main__':
    # Load the data
    main_dir = '/root/CodeHub/py/radar-pol-wforcast/.data/new_2308_1'
    altitude = '1.0km'
    datasets = dl.load_data(main_dir, altitude, maxsize=5)
    print(np.array(datasets[0]).shape)
    load_xy(datasets, sample_dir_size=1, normalparam=normalparam, split_ratio=0.8, window_size=10, stride=1)
else:
    pass


