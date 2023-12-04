import numpy as np
from scipy.linalg import norm
import dataload.dataloadv4 as dl
from pykalman import KalmanFilter

def slide_windows(data, tstep=10, overlap=0.5):
    seqs = []
    lendata = len(data[0])
    move = int((1 - overlap) * tstep)
    # use slide windows
    for i in range(0, lendata - tstep + 1, move):
        seq = data[:, i:i+tstep, :, :]
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
                            (filtered_state_means, _) = kf.filter_update(
                                filtered_state_means[-1], kf.transition_matrix, observation
                            )
                            kf = kf.filter_update(filtered_state_means[-1], kf.transition_matrix, observation)

                        filtered_data[i, t, j, k, channel] = filtered_state_means.flatten()

    return filtered_data

kalman_filter = kalman_filter_iterative

def load_xy(datasets, idr, window_size, overlap, normalparam):
    samples = np.array([]).reshape(0, 3, window_size, 256, 256)
    for id in idr:
        samples = np.concatenate((samples, np.array(slide_windows(np.array(datasets[id]), window_size, overlap=0.88))), axis=0)
    samples = samples.swapaxes(1, 2)
    samples = samples.swapaxes(2, 3)
    samples = samples.swapaxes(3, 4)
    
    # normalization 
    for x in range(3):
        mmin, mmax = normalparam[x]
        samples[:, :, :, :, x] = (samples[:, :, :, :, x] - mmin) / (mmax - mmin)
    
    # make X and y
    stride = int(window_size * (1 - overlap))
    X = samples[:-window_size, :, :, :, :]
    y = samples[window_size:, :, :, :, 0:1]
    
    # split train and test
    split_ratio = 0.8
    X_train, X_test, y_train, y_test = split_dataset(X, y, split_ratio)
    
    print(f'X_train.shape: {X_train.shape}')
    print(f'X_test.shape: {X_test.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'y_test.shape: {y_test.shape}')
    return X_train, X_test, y_train, y_test




# Load the data
main_dir = '/root/CodeHub/py/radar-pol-wforcast/.data/new_2308_1'
altitude = '1.0km'
datasets = dl.load_data(main_dir, altitude)
norm_param = ([0, 65], [-1, 5], [-1, 6])  # dBZ: [0, 65], ZDR: [-1, 5], KDP: [-1, 6]

if __name__ == '__main__':
    load_xy(datasets, range(5), 10, 0.88, norm_param)
else:
    pass


