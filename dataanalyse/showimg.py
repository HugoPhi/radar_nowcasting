import tkinter as tk
import os
from PIL import Image
import matplotlib.pyplot as plt

def display_images(file_paths):
    for file_path in file_paths:
        # 使用Pillow加载图像数据
        image = Image.open(file_path)

        # 使用matplotlib显示图像
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title(file_path)

        def on_key(event):
            # 处理按键事件
            if event.key == ' ':
                # 退出全屏
                fig.canvas.manager.window.attributes('-fullscreen', False)
                plt.close()

        # 连接键盘事件处理器
        fig.canvas.mpl_connect('key_press_event', on_key)

        # 全屏显示图像
        fig.canvas.manager.window.attributes('-fullscreen', True)

        # 显示图像
        plt.show()

i, j = 1, 2
# 从外部传入文件路径并显示图像
image_paths = ["/root/CodeHub/py/radar-pol-wforcast/dataanalyse/figures/1_1/" + x for x in sorted(os.listdir(f"/root/CodeHub/py/radar-pol-wforcast/dataanalyse/figures/{i}_{j}/"))]
display_images(image_paths)

