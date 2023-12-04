import os
import numpy as np

def load_data(dataset_root, variable, altitude, begin_frame, end_frame):
    data = []

    for frame_number in range(begin_frame, end_frame + 1):
        frame_str = f"{frame_number:03d}"
        data_dir_path = os.path.join(dataset_root, variable, altitude, f"data_dir_{frame_str}")
        
        for file_name in os.listdir(data_dir_path):
            if file_name.endswith('.npy') and file_name.startswith('frame'):
                file_path = os.path.join(data_dir_path, file_name)
                loaded_data = np.load(file_path)
                data.append(loaded_data)

    return np.concatenate(data, axis=0)

# 数据集根目录
dataset_root = '/root/CodeHub/py/radar-pol-wforcast/.data/new_2308_1'

# 要加载的范围
begin_frame = 1
end_frame = 1

# 加载数据
kdp = load_data(dataset_root, 'KDP', '1.0km', begin_frame, end_frame)
dbz = load_data(dataset_root, 'dBZ', '1.0km', begin_frame, end_frame)
zdr = load_data(dataset_root, 'ZDR', '1.0km', begin_frame, end_frame)

# 输出加载后的数据形状
print(f'KDP shape: {kdp.shape}')
print(f'dBZ shape: {dbz.shape}')
print(f'ZDR shape: {zdr.shape}')

