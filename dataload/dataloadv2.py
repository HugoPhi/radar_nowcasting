import os
import numpy as np

def load_data(dataset_root, variable, altitudes, begin_frame, end_frame):
    data = []

    for altitude in altitudes:
        altitude_data = []
        for frame_number in range(begin_frame, end_frame + 1):
            frame_str = f"{frame_number:03d}"
            data_dir_path = os.path.join(dataset_root, variable, altitude, f"data_dir_{frame_str}")

            for file_name in os.listdir(data_dir_path):
                if file_name.endswith('.npy') and file_name.startswith('frame'):
                    file_path = os.path.join(data_dir_path, file_name)
                    loaded_data = np.load(file_path)
                    altitude_data.append(loaded_data)

        data.append(np.concatenate(altitude_data, axis=0))

    return np.stack(data, axis=0)


gmean = 0
gstd = 1

def normalize_data(data):
    gmean = np.mean(data)
    gstd = np.std(data)
    normalized_data = (data - gmean) / gstd
    return normalized_data

# 数据集根目录
dataset_root = '/root/CodeHub/py/radar-pol-wforcast/.data/new_2308_1'

# 要加载的范围
begin_frame = 1
end_frame = 2

# 高度信息列表
altitudes = ['1.0km', '3.0km', '7.0km']

# 加载数据
kdp = load_data(dataset_root, 'KDP', altitudes, begin_frame, end_frame)
dbz = load_data(dataset_root, 'dBZ', altitudes, begin_frame, end_frame)
zdr = load_data(dataset_root, 'ZDR', altitudes, begin_frame, end_frame)

# 对数据进行零均值和归一化
kdp_normalized = normalize_data(kdp)
dbz_normalized = normalize_data(dbz)
zdr_normalized = normalize_data(zdr)

# 输出加载后的数据形状
print(f'KDP normalized shape: {kdp_normalized.shape}')
print(f'dBZ normalized shape: {dbz_normalized.shape}')
print(f'ZDR normalized shape: {zdr_normalized.shape}')


# 逆变换：
def turn_back(data):
    turn_back_data = data * gstd + gmean
    return turn_back_data



