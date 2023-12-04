import os
import matplotlib.pyplot as plt
import dataload.dataloadv4 as dl
from matplotlib import gridspec

max_num = 5  # data_dir 的数量


# 假设你已经有了相应的数据，这里使用随机数据作为示例
data_shape = (256, 256)
depths = ['1.0km', '3.0km', '7.0km']
variables = ['KDP', 'ZDR', 'dBZ']


def draw(idi, idj):
    # 随机生成KDP、ZDR、dBZ数据
    kdp_data, zdr_data, dbz_data = dl.load_data('/root/CodeHub/py/radar-pol-wforcast/.data/new_2308_1', ('1.0km', '3.0km', '7.0km'), idi, idj)
    text_frequency = 32  # 调整文本显示的频率

    for i, var_name in enumerate(variables):
        for j, depth in enumerate(depths):
            # 创建一个子图布局
            fig, ax = plt.subplots(figsize=(9, 9))

            # 绘制子图
            img = ax.imshow(locals()[f'{var_name.lower()}_data'][j], cmap='viridis', extent=[0, 256, 0, 256])

            # 添加颜色条
            cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.1)
            cbar.set_label('Value')

            # 设置子图标题和坐标轴标签
            ax.set_title(f'{var_name} at {depth}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            # 显示深度对应的值的大小（减少显示频率）
            # for x in range(0, 256, text_frequency):
            #     for y in range(0, 256, text_frequency):
            #         ax.text(x, y, f'{locals()[f"{var_name.lower()}_data"][j, x, y]:.2f}', ha='center', va='center', fontsize=6, color='white')
             
            # 保存子图
            os.makedirs(f'./figures/{idi}_{idj}', exist_ok=True)
            plt.savefig(f'./figures/{idi}_{idj}/{var_name}_{depth}.png')
            plt.close()

    # 如果你需要分开显示图形，可以在这里添加显示图形的代码，例如：
    # plt.show()

for i in range(1, 6):
    for j in range(len(os.listdir(f'/root/CodeHub/py/radar-pol-wforcast/.data/new_2308_1/KDP/1.0km/data_dir_{i:03d}'))):
        draw(i, j)

