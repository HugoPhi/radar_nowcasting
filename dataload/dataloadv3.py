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


# 归一化数据
mean = {'KDP': 0.5, 'dBZ': 30.0, 'ZDR': 2.0}  # 根据实际数据调整
std = {'KDP': 1.0, 'dBZ': 10.0, 'ZDR': 1.0}    # 根据实际数据调整

## 1 计算均值和标准差
mean['KDP'] = np.mean(kdp)
std['KDP'] = np.std(kdp)

mean['dBZ'] = np.mean(dbz)
std['dBZ'] = np.std(dbz)

mean['ZDR'] = np.mean(zdr)
std['ZDR'] = np.std(zdr)

## 2 归一化
kdp_normalized = (kdp - mean['KDP']) / std['KDP']
dbz_normalized = (dbz - mean['dBZ']) / std['dBZ']
zdr_normalized = (zdr - mean['ZDR']) / std['ZDR']


# 输出加载后的数据形状
print(f'KDP normalized shape: {kdp_normalized.shape}')
print(f'dBZ normalized shape: {dbz_normalized.shape}')
print(f'ZDR normalized shape: {zdr_normalized.shape}')

def reverse_normalize(data):
    predicted_dbz = data.squeeze() * std['dBZ'] + mean['dBZ']


