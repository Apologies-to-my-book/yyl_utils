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
                 window_size: int = 10000, figsize=(12, 6)):
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
        from matplotlib.widgets import Slider, TextBox, Button
        import numpy as np

        self.data = data
        self.time_stamp = time_stamp
        self.fs = np.mean(np.diff(self.time_stamp))
        self.window_size = window_size
        self.current_position = 0
        self.ylim = (-0.4, 0.4)  # 默认Y轴范围

        self.fig = plt.figure(figsize=figsize)

        # 使用 GridSpec 划分上下图区域（高度比 1:2 或 1:3）
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.35,
                                   left=0.08, right=0.92, top=0.95, bottom=0.12)

        self.ax_overview = self.fig.add_subplot(gs[0])
        self.ax_detail = self.fig.add_subplot(gs[1])
        # 取消显示科学计数法
        import matplotlib.ticker as ticker
        for ax in [self.ax_overview, self.ax_detail]:
            # X轴
            formatter_x = ticker.ScalarFormatter(useOffset=False)
            formatter_x.set_scientific(False)
            ax.xaxis.set_major_formatter(formatter_x)
            # Y轴
            formatter_y = ticker.ScalarFormatter(useOffset=False)
            formatter_y.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter_y)

        # 概览图（降采样）
        self.ax_overview.plot(
            self.resample_data(self.time_stamp, points_overview),
            self.resample_data(self.data, points_overview),
            'g', alpha=0.7, linewidth=0.5
        )
        self.ax_overview.set_title('overview')
        self.ax_overview.set_ylabel('value')
        self.ax_overview.set_ylim(-0.5, 0.5)  # 设置概览图Y轴范围
        # 只在上方图显示 x 轴标签，下方图也显示也不冲突，
        # 但为了简洁可只在一个地方显示或都不显示
        # self.ax_overview.set_xlabel('')

        # 详细图初始线
        end = self.current_position + self.window_size
        self.detail_line, = self.ax_detail.plot(
            self.time_stamp[self.current_position:end],
            self.data[self.current_position:end],
            'r', linewidth=1
        )
        self.ax_detail.set_ylim(self.ylim)
        self.ax_detail.set_title('detail')
        self.ax_detail.set_xlabel('time_stamp')
        self.ax_detail.set_ylabel('value')

        # 滑块和按钮
        slider_ax = self.fig.add_axes([0.15, 0.02, 0.65, 0.03])
        self.slider = Slider(slider_ax, 'Position', 0,
                             len(data) - self.window_size - 1,
                             valinit=self.current_position, valfmt='%d')
        self.slider.on_changed(self.on_slider_change)

        # 左按钮（后退）
        btn_left_ax = self.fig.add_axes([0.08, 0.02, 0.04, 0.03])
        self.btn_left = Button(btn_left_ax, '<')

        # 右按钮（前进）
        btn_right_ax = self.fig.add_axes([0.82, 0.02, 0.04, 0.03])
        self.btn_right = Button(btn_right_ax, '>')

        # 按钮长按相关
        self.press_timer = None
        self.press_direction = 0
        self.step_size = 2*int(1/self.fs)
        self.btn_left_ax = btn_left_ax
        self.btn_right_ax = btn_right_ax
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        # 文本框
        textbox_ax = self.fig.add_axes([0.92, 0.05, 0.06, 0.03])
        self.textbox = TextBox(textbox_ax, 'Jump: ',
                               initial=str(self.current_position))
        self.textbox.on_submit(self.on_text_submit)

        ylim_ax = self.fig.add_axes([0.92, 0.09, 0.06, 0.03])
        self.ylim_textbox = TextBox(ylim_ax, 'YLim: ', initial=f'{self.ylim[0]},{self.ylim[1]}')
        self.ylim_textbox.on_submit(self.on_ylim_submit)

        # 概览图上的位置标记
        self.position_marker = self.ax_overview.axvspan(
            self.time_stamp[self.current_position],
            self.time_stamp[min(end, len(self.data) - 1)],
            alpha=0.3, color='g'
        )


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

    def on_mouse_press(self, event):
        """鼠标按下事件 - 检测是否在按钮区域"""
        if event.inaxes == self.btn_left_ax:
            self.press_direction = -1
            self._move_position()
            if self.press_timer is None:
                self.press_timer = self.fig.canvas.new_timer(interval=10)  # 10ms间隔
                self.press_timer.add_callback(self._timer_move)
                self.press_timer.start()
        elif event.inaxes == self.btn_right_ax:
            self.press_direction = 1
            self._move_position()
            if self.press_timer is None:
                self.press_timer = self.fig.canvas.new_timer(interval=10)
                self.press_timer.add_callback(self._timer_move)
                self.press_timer.start()

    def on_mouse_release(self, event):
        """鼠标释放事件"""
        if self.press_timer is not None:
            self.press_timer.stop()
            self.press_timer = None
        self.press_direction = 0

    def _timer_move(self):
        """定时器回调 - 长按时连续移动"""
        self._move_position()

    def _move_position(self):
        """移动位置的核心逻辑"""
        if self.press_direction == -1:
            step = max(0, self.current_position - self.step_size)
        elif self.press_direction == 1:
            max_position = len(self.data) - self.window_size - 1
            step = min(max_position, self.current_position + self.step_size)
        else:
            return

        self.current_position = step
        self.slider.set_val(step)
        self.update_display()

    def on_ylim_submit(self, text):
        try:
            parts = text.split(',')
            ymin = float(parts[0].strip())
            ymax = float(parts[1].strip())
            if ymin < ymax:
                self.ylim = (ymin, ymax)
                self.ax_detail.set_ylim(self.ylim)
                self.fig.canvas.draw_idle()
            else:
                print("ymin must be less than ymax!")
        except (ValueError, IndexError):
            print("Invalid format! Use: min,max (e.g., -0.1,0.1)")

    def update_display(self):
        """更新所有显示内容"""
        import numpy as np
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
        self.ax_detail.set_ylim(self.ylim)

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
        # 强制取消科学计数法（放在 draw 之前）
        import matplotlib.ticker as ticker

        for ax in [self.ax_overview, self.ax_detail]:
            # X轴
            formatter_x = ticker.ScalarFormatter(useOffset=False)
            formatter_x.set_scientific(False)
            ax.xaxis.set_major_formatter(formatter_x)
            # Y轴
            formatter_y = ticker.ScalarFormatter(useOffset=False)
            formatter_y.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter_y)

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