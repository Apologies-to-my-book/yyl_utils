from __future__ import annotations
import matplotlib.font_manager as fm
import os
from typing import Union,Iterable
import pandas as pd
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

class LargeDataSlidePlot:
    def __init__(self, data: np.ndarray, time_stamp: np.ndarray, points_overview: int = 30000,
                 window_size: int = 10000, figsize=(14, 8)):
        """
        大数据滑动绘图类

        Args:
            data: 数据数组
            time_stamp: 时间戳数组
            points_overview: 概览图显示的点数，默认30000点
            window_size: 详细图窗口大小，默认30000点
            figsize: 图形尺寸
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, TextBox

        self.data = data
        self.time_stamp = time_stamp
        self.fig = plt.figure(figsize=figsize)

        # 创建两个子图：上方用于概览，下方用于详细显示
        # 坐标参数: [左边距, 底边距, 宽度, 高度] (相对坐标0-1)
        self.ax_overview = plt.axes([0.1, 0.7, 0.8, 0.2])  # 上方概览图
        self.ax_detail = plt.axes([0.1, 0.1, 0.8, 0.5])  # 下方详细图

        # 初始参数
        self.window_size = window_size  # 详细图显示的数据点数
        self.current_position = 0  # 当前显示的数据起始位置

        # 绘制概览图 - 显示数据的整体趋势
        self.ax_overview.plot(
            self.resample_data(self.time_stamp, points_overview),  # 降采样时间戳
            self.resample_data(self.data, points_overview),  # 降采样数据
            'g', alpha=0.7, linewidth=0.5  # 黑色细线，半透明
        )
        self.ax_overview.set_title('overview_figure')
        self.ax_overview.set_ylabel("value")
        self.ax_overview.set_xlabel("time_stamp")

        # 绘制详细图 - 显示当前窗口的详细数据
        # 保存Line2D对象引用，便于后续更新
        self.detail_line, = self.ax_detail.plot(
            self.time_stamp[self.current_position:self.current_position + self.window_size],  # 当前窗口的时间戳
            self.data[self.current_position:self.current_position + self.window_size],  # 当前窗口的数据
            'r', linewidth=1  # 绿色细线
        )
        self.ax_detail.set_title('detail_figure')
        self.ax_detail.set_xlabel('time_stamp')
        self.ax_detail.set_ylabel("value")

        # 创建滑块 - 用于控制详细图的显示位置
        self.slider_ax = plt.axes([0.1, 0.02, 0.8, 0.03])  # 滑块位置
        self.slider = Slider(
            self.slider_ax,  # 滑块放置的坐标轴
            'positions',  # 滑块标签
            0,  # 最小值
            len(data) - self.window_size - 1,  # 最大值（防止数组越界）
            valinit=self.current_position,  # 初始值
            valfmt='%d'  # 显示格式：整数
        )
        self.slider.on_changed(self.on_slider_change)  # 绑定滑块变化回调函数

        # 添加文本框输入 - 用于直接输入位置
        self.textbox_ax = plt.axes([0.95, 0.05, 0.04, 0.02])  # 文本框位置
        self.textbox = TextBox(
            self.textbox_ax,
            'jump: ',
            initial=str(self.current_position)
        )
        self.textbox.on_submit(self.on_text_submit)  # 绑定文本框提交事件

        # 在概览图上添加位置标记 - 显示当前详细图在整体数据中的位置
        self.position_marker = self.ax_overview.axvspan(
            self.time_stamp[self.current_position],  # 起始时间
            self.time_stamp[self.current_position + self.window_size],  # 结束时间
            alpha=0.3, color='g'  # 绿色半透明区域
        )

        # 初始显示更新
        self.update_display()

    @staticmethod
    def resample_data(data, num_points):
        """
        数据降采样方法

        Args:
            data: 原始数据数组
            num_points: 目标点数

        Returns:
            降采样后的数据
        """
        n_step = len(data) // num_points  # 计算步长
        return data[::n_step]  # 按步长抽取数据点

    def on_text_submit(self, text):
        """
        文本框提交回调函数 - 处理直接输入的位置

        Args:
            text: 输入的文本内容
        """
        try:
            # 将输入转换为整数
            target_position = int(text)

            # 验证输入范围
            max_position = len(self.data) - self.window_size - 1
            if 0 <= target_position <= max_position:
                self.current_position = target_position
                # 更新滑块位置
                self.slider.set_val(target_position)
                # 更新显示
                self.update_display()
            else:
                print(f"输入位置超出范围! 有效范围: 0 - {max_position}")

        except ValueError:
            print("请输入有效的数字!")

    def on_slider_change(self, val):
        """
        滑块回调函数 - 处理滑块值变化

        Args:
            val: 滑块当前值（数据起始位置）
        """
        self.current_position = int(val)  # 更新当前位置
        self.update_display()  # 更新显示

    def update_display(self):
        """更新所有显示内容"""
        # 计算显示范围的结束索引，防止数组越界
        end_idx = min(self.current_position + self.window_size, len(self.data) - 1)

        # 提取当前窗口的数据
        x_data = self.time_stamp[self.current_position:end_idx]  # 时间戳数据
        y_data = self.data[self.current_position:end_idx]  # 数值数据

        # 边界处理：如果数据长度不足窗口大小，用NaN填充
        if len(y_data) < self.window_size:
            y_data = np.pad(y_data, (0, self.window_size - len(y_data)),
                            mode='constant', constant_values=np.nan)

        # 更新详细图数据
        self.detail_line.set_data(x_data, y_data)

        # 设置详细图的坐标轴范围
        self.ax_detail.set_xlim(self.time_stamp[self.current_position],  # X轴起始时间
                                self.time_stamp[end_idx])  # X轴结束时间
        self.ax_detail.set_ylim(np.nanmin(y_data),
                                np.nanmax(y_data))

        # 更新概览图中的位置标记
        self.position_marker.remove()  # 删除旧的位置标记
        # 创建新的位置标记
        self.position_marker = self.ax_overview.axvspan(
            self.time_stamp[self.current_position],  # 起始时间
            self.time_stamp[end_idx],  # 结束时间
            alpha=0.3, color='r', label='detail_area'  # 绿色半透明区域，带标签
        )

        # 更新标题显示信息
        self.ax_overview.set_title(f'overview - total_points: {len(self.data):,} | view {self.window_size} points')
        self.ax_detail.set_title(
            f'detail_figure: {self.time_stamp[self.current_position]}---{self.time_stamp[end_idx]}')

        # 添加图例
        self.ax_overview.legend()

        # 重绘图形，显示所有更新
        self.fig.canvas.draw_idle()

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()

    def close(self):
        import matplotlib.pyplot as plt
        plt.close()


def fast_plot_psd(lfp_data: np.ndarray,
                 freq: Iterable[Union[float, int]] = (0, 100),
                 fs: float = 1000,
                 nperseg: int = 1024,
                 scaling: str = 'density',
                 psd_unit: str = 'raw',
                color:str = 'k',
                  ) -> tuple:
    """
    计算并绘制LFP数据的功率谱密度，注意welch默认返回的单边谱就已经是乘以2的结果了，不用再乘了

    参数:
    lfp_data: 输入的LFP数据，应为1维数组
    freq: 频率范围，包含两个元素的迭代对象 [低频, 高频]，默认(0, 100)Hz
    fs: 采样频率 (Hz)，默认1000Hz
    nperseg: Welch方法中每个段的长度，默认1024
    scaling: Welch方法返回的是psd还是功率，注意psd/功率是幅值的平方
    psd_unit: 功率谱密度单位，'raw'表示线性坐标，'dB'表示分贝坐标

    返回:
    tuple: (频率数组, 功率谱密度数组)

    异常:
    ValueError: 当输入数据不为1维或psd_unit参数无效时
    """
    from scipy import signal
    import matplotlib.pyplot as plt
    # 检查输入数据维度
    if len(lfp_data.shape) > 1:
        raise ValueError("输入数据不为1维，请提供单通道LFP数据")

    # 检查psd_unit参数有效性 - 修复了原来的逻辑错误
    if psd_unit not in ['raw', 'dB']:
        raise ValueError("psd_unit必须为 'raw' 或 'dB'")

    # 检查频率范围参数
    if len(freq) != 2:
        raise ValueError("freq参数必须包含2个元素 [低频, 高频]")

    low_freq, high_freq = freq
    if low_freq >= high_freq:
        raise ValueError("频率范围无效：低频必须小于高频")

    # 计算功率谱密度
    f, psd = signal.welch(lfp_data,
                          fs=fs,
                          nperseg=nperseg,
                          scaling=scaling)

    # 选择指定频率范围
    mask = (f >= low_freq) & (f <= high_freq)
    # noinspection PyUnresolvedReferences
    f_mask = f[mask]
    # noinspection PyUnresolvedReferences
    psd_mask = psd[mask]

    if psd_unit == 'raw':
        # 绘制线性坐标的PSD
        plt.plot(f_mask, psd_mask, linewidth=1.5,color=color)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectrum Density (V²/Hz)')
        plt.title('Power Spectrum Density of LFP Data')
        plt.grid(True, alpha=0.3)
    elif psd_unit == 'dB':
        # 绘制分贝坐标的PSD
        # 避免对0或负值取对数
        psd_dB = 10 * np.log10(np.maximum(psd_mask, 1e-10))
        plt.plot(f_mask, psd_dB, linewidth=1.5,color=color)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectrum Density (dB)')
        plt.title('Power Spectrum Density of LFP Data (dB scale)')
        plt.grid(True, alpha=0.3)

    # 设置x轴范围，在频率范围两侧留出10%的边距
    freq_range = high_freq - low_freq
    plt.xlim([low_freq - freq_range * 0.1,
              high_freq + freq_range * 0.1])

    return f, psd

def get_color_with_opencv(image_path):
    """
    取色器函数，输入image_path是图片路径，输入后会生成一个图，在图中点击对应色块会显示该色块的颜色编号
    Args:
        image_path:

    Returns:

    """
    import cv2
    # 读取图片
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 获取颜色（OpenCV是BGR格式）
            b, g, r = img[y, x]

            # 转换为十六进制
            hex_color = f'#{r:02x}{g:02x}{b:02x}'

            print(f'位置: ({x}, {y})')
            print(f'BGR: ({b}, {g}, {r})')
            print(f'RGB: ({r}, {g}, {b})')
            print(f'HEX: {hex_color}')
            print('-' * 30)

            # 在图片上显示颜色信息
            display_img = img.copy()
            cv2.putText(display_img, hex_color, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Image', display_img)

    # 显示图片并设置鼠标回调
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mouse_callback)

    print("点击图片获取颜色，按ESC退出")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            break

    cv2.destroyAllWindows()

def set_backend(backend_name='Qt5Agg'):
    '''
    必须在import matplotlib.pyplot as plt前设置，print(matplotlib.get_backend())查看后端
    常用后端：
    # GUI 后端
        'TkAgg'        # Tkinter (最常用)
        'Qt5Agg'       # PyQt5
        'Qt4Agg'       # PyQt4
        'GTK3Agg'      # GTK+ 3
        'GTK4Agg'      # GTK+ 4
        'WXAgg'        # wxPython
        'MacOSX'       # macOS 原生
    # 静态图像后端
        'agg'          # 抗锯齿 PNG (默认)
        'pdf'          # PDF 输出
        'svg'          # SVG 矢量图
        'ps'           # PostScript
        'cairo'        # Cairo 图形
    :param backend_name:设置后端名
    '''
    os.environ['MPLBACKEND'] = backend_name  # 通过环境变量设置

def set_chinese_font():
    import matplotlib.pyplot as plt
    # 常见的中文字体名称
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'SimSun']
    for font in chinese_fonts:
        if any(f.name == font for f in fm.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用字体: {font}")
            return True
    print("未找到中文字体，请安装中文字体")
    return False

def plot_CI_map(list_group_data:list[np.ndarray],x_array:np.ndarray,group_name:Union[list[str],None]=None):
    """
        绘制多组数据的置信区间线图

        Args:
            list_group_data: 每组数据的数组列表，每个数组形状为 (样本数, x点数)
            x_array: x轴坐标数组
            group_name: 组名列表，如果为None则自动生成
        """
    import seaborn as sns
    num_groups=len(list_group_data)
    if not group_name:
        group_name=np.arange(num_groups).astype(str)
    dict_all_data = {'x': [], 'y': [], 'group_name': []}
    for iter,single_group_array in enumerate(list_group_data):
        if len(single_group_array.shape)==1:
            single_group_array=single_group_array.reshape(-1,1)
        num_samples=single_group_array.shape[0]
        dict_all_data['x'].extend(np.repeat(x_array,num_samples))
        dict_all_data['y'].extend(single_group_array.flatten('C'))
        dict_all_data['group_name'].extend([group_name[iter]]*len(x_array)*num_samples)
    df = pd.DataFrame(dict_all_data)
    sns.lineplot(data=df, x='x', y='y', hue='group_name',
                 errorbar=('ci', 95),
                 err_style='band',
                 # palette=['#FF6B6B', '#4ECDC4'],  # 自定义颜色
                 linewidth=2.5,
                 err_kws={'alpha': 0.35})