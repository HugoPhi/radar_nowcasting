import os
import shutil

def create_selected_dataset(dataset_root, selected_range, output_dataset_root):
    for variable in os.listdir(dataset_root):
        variable_path = os.path.join(dataset_root, variable)
        if os.path.isdir(variable_path):
            for altitude in os.listdir(variable_path):
                altitude_path = os.path.join(variable_path, altitude)
                if os.path.isdir(altitude_path):
                    for data_dir_number in range(selected_range[0], selected_range[1] + 1):
                        data_dir_str = f"data_dir_{data_dir_number:03d}"
                        src_data_dir_path = os.path.join(dataset_root, variable, altitude, data_dir_str)
                        dst_data_dir_path = os.path.join(output_dataset_root, variable, altitude, data_dir_str)
                        shutil.copytree(src_data_dir_path, dst_data_dir_path)

# 数据集根目录
dataset_root = 'path/to/NJU_CPOL_update2308'

# 要选择的部分数据目录范围，例如选择 data_dir_001 到 data_dir_005
selected_range = (1, 5)

# 新数据集的输出路径
output_dataset_root = dataset_root + '/new'

# 在新数据集的根目录下创建相应的目录结构
for variable in os.listdir(os.path.join(dataset_root)):
    variable_path = os.path.join(dataset_root, variable)
    if os.path.isdir(variable_path):
        for altitude in os.listdir(variable_path):
            altitude_path = os.path.join(variable_path, altitude)
            if os.path.isdir(altitude_path):
                os.makedirs(os.path.join(output_dataset_root, variable, altitude), exist_ok=True)

# 调用函数创建新数据集
create_selected_dataset(dataset_root, selected_range, output_dataset_root)
# 指定压缩后的文件名（不包含扩展名）
output_archive_name = dataset_root + f'new_{selected_range[0]}_{selected_range[1]}'
shutil.make_archive(output_archive_name, 'zip', output_dataset_root)
