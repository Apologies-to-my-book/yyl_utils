from __future__ import annotations
from typing import Iterable,Union


def fit_log_pink_noise(freqs, psd, fit_range=(1, 40), peak_width_limits=(0.98 * 2, 8),
                       max_n_peaks=6, min_peak_height=0.1, verbose=True):
    """
    拟合功率谱，分离背景1/f噪声和振荡峰值

    Parameters
    ----------
    freqs : array_like
        频率数组 (Hz)
    psd : array_like
        功率谱密度
    fit_range : tuple, default=(1, 40)
        拟合频率范围 [低, 高]
    peak_width_limits : tuple, default=(0.98*2, 8)
        峰值宽度限制 [最小, 最大] (Hz)
    max_n_peaks : int, default=6
        最多检测的峰值数
    min_peak_height : float, default=0.1
        最小峰值高度
    verbose : bool, default=True
        是否打印和绘图

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        返回三个数组：
        freq_mask : ndarray - 指定频率范围的频率
        psd_mask : ndarray - 原始PSD（指定范围）
        psd_fit_mask : ndarray - 拟合的背景噪声PSD（指定范围）
    """
    from specparam import SpectralModel
    import numpy as np
    # 初始化并拟合模型
    fm = SpectralModel(peak_width_limits=peak_width_limits,
                       max_n_peaks=max_n_peaks,
                       min_peak_height=min_peak_height,
                       verbose=verbose)
    fm.fit(freqs, psd, fit_range)

    if verbose:
        fm.print_results()
        fm.plot()

    # 提取指定范围的背景拟合结果
    mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    freq_mask = freqs[mask]
    psd_mask = psd[mask]

    # 计算背景拟合：10^(offset - slope * log10(freq))
    offset, slope = fm.results.params.aperiodic.params[0:2]
    psd_fit_mask = 10 ** (offset - slope * np.log10(freq_mask))

    return freq_mask, psd_mask, psd_fit_mask


class SOSFilter:
    """
    SOS滤波器类，支持低通、高通、带通、陷波滤波器
    使用二阶分节形式和filtfilt双向滤波确保零相位延迟
    """
    def __init__(self, filter_type, fs, order=4, filter_design='butter',
                 f1=None, f2=None, notch_freq=None, notch_Q=30, rp=1, rs=40,
                 padlen=None):
        """
        初始化滤波器

        参数:
        filter_type : str
            滤波器类型: 'lowpass', 'highpass', 'bandpass', 'bandstop', 'notch'
        fs : float
            采样频率 (Hz)
        order : int
            滤波器阶数
        filter_design : str
            滤波器设计方法: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'
        f1 : float, optional
            对于低通/高通：截止频率；对于带通/带阻：低频截止频率
        f2 : float, optional
            对于带通/带阻：高频截止频率
        notch_freq : float, optional
            陷波频率 (仅用于notch类型)
        notch_Q : float, optional
            陷波滤波器的Q值 (默认: 30)
        rp : float, optional
            通带纹波 (dB)，用于cheby1和ellip
        rs : float, optional
            阻带衰减 (dB)，用于cheby2和ellip
        padlen : int, optional
            filtfilt的填充长度，None时自动计算
        """
        global np, signal, plt, yyl
        # noinspection PyUnresolvedReferences
        import numpy as np
        # noinspection PyUnresolvedReferences
        from scipy import signal
        # noinspection PyUnresolvedReferences
        import matplotlib.pyplot as plt
        self.filter_type = filter_type.lower()
        self.fs = fs
        self.order = order
        self.filter_design = filter_design.lower()
        self.f1 = f1
        self.f2 = f2
        self.notch_freq = notch_freq
        self.notch_Q = notch_Q
        self.rp = rp
        self.rs = rs
        self.padlen = padlen
        self.sos = None

        self.design_filter()

    def design_filter(self):
        """设计SOS滤波器"""
        try:
            nyquist = 0.5 * self.fs

            if self.filter_type == 'notch':
                # 陷波滤波器特殊处理
                self._design_notch_filter()
            else:
                # 标准IIR滤波器
                self._design_standard_filter(nyquist)

            print(f"成功设计 {self.filter_design} {self.filter_type} 滤波器")
            if self.filter_type in ['lowpass', 'highpass']:
                print(f"截止频率: {self.f1} Hz, 阶数: {self.order}")
            elif self.filter_type in ['bandpass', 'bandstop']:
                print(f"频率范围: {self.f1}-{self.f2} Hz, 阶数: {self.order}")
            elif self.filter_type == 'notch':
                print(f"陷波频率: {self.notch_freq} Hz, Q值: {self.notch_Q}")

        except Exception as e:
            print(f"滤波器设计失败: {e}")
            self.sos = None

    def _design_standard_filter(self, nyquist):
        """设计标准IIR滤波器"""
        if self.filter_type == 'lowpass':
            cutoff = self.f1 / nyquist
            Wn = cutoff
        elif self.filter_type == 'highpass':
            cutoff = self.f1 / nyquist
            Wn = cutoff
        elif self.filter_type == 'bandpass':
            low = self.f1 / nyquist
            high = self.f2 / nyquist
            Wn = [low, high]
        elif self.filter_type == 'bandstop':
            low = self.f1 / nyquist
            high = self.f2 / nyquist
            Wn = [low, high]
        else:
            raise ValueError(f"不支持的滤波器类型: {self.filter_type}")

        btype = self.filter_type

        # 根据设计方法选择对应的函数
        if self.filter_design == 'butter':
            self.sos = signal.butter(self.order, Wn, btype=btype, output='sos')
        elif self.filter_design == 'cheby1':
            self.sos = signal.cheby1(self.order, self.rp, Wn, btype=btype, output='sos')
        elif self.filter_design == 'cheby2':
            self.sos = signal.cheby2(self.order, self.rs, Wn, btype=btype, output='sos')
        elif self.filter_design == 'ellip':
            self.sos = signal.ellip(self.order, self.rp, self.rs, Wn, btype=btype, output='sos')
        elif self.filter_design == 'bessel':
            self.sos = signal.bessel(self.order, Wn, btype=btype, output='sos')
        else:
            raise ValueError(f"不支持的滤波器设计方法: {self.filter_design}")

    def _design_notch_filter(self):
        """设计陷波滤波器"""
        if self.notch_freq is None:
            raise ValueError("陷波滤波器需要指定notch_freq参数")

        # 计算归一化频率
        w0 = self.notch_freq / (0.5 * self.fs)

        # iirnotch 返回 (b, a)
        b, a = signal.iirnotch(w0, self.notch_Q)

        # 转换为 SOS 格式
        self.sos = signal.tf2sos(b, a)

    def filtfilt(self, data, padlen=None):
        """
        使用filtfilt进行双向滤波（零相位延迟）

        参数:
        data : array_like
            输入数据
        padlen : int, optional
            填充长度，None时使用self.padlen或自动计算

        返回:
        filtered_data : ndarray
            滤波后的数据（零相位延迟）
        """
        if self.sos is None:
            raise ValueError("滤波器未正确设计")

        # 确定填充长度
        if padlen is None:
            padlen = self.padlen

        # 使用sosfiltfilt进行双向滤波
        filtered_data = signal.sosfiltfilt(self.sos, data, padlen=padlen)
        return filtered_data

    def filter_forward(self, data, zi=None):
        """
        前向滤波（会导致相位的非线性响应，不要使用）

        参数:
        data : array_like
            输入数据
        zi : array_like, optional
            初始条件

        返回:
        filtered_data : ndarray
            滤波后的数据
        zf : ndarray
            最终状态（用于连续滤波）
        """
        if self.sos is None:
            raise ValueError("滤波器未正确设计")

        if zi is None:
            # 自动计算初始条件
            if len(data) > 0:
                zi = signal.sosfilt_zi(self.sos) * data[0]
            else:
                zi = signal.sosfilt_zi(self.sos)

        filtered_data, zf = signal.sosfilt(self.sos, data, zi=zi)
        return filtered_data, zf

    def plot_response(self, worN=2000, freq_lim=None):
        """绘制频率响应"""
        if self.sos is None:
            print("滤波器未设计")
            return

        w, h = signal.sosfreqz(self.sos, worN=worN, fs=self.fs)

        plt.figure(figsize=(12, 8))

        # 幅度响应
        plt.subplot(2, 1, 1)
        plt.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))

        # 根据滤波器类型添加参考线
        if self.filter_type == 'lowpass':
            plt.axvline(self.f1, color='red', linestyle='--', alpha=0.7, label=f'截止 {self.f1} Hz')
        elif self.filter_type == 'highpass':
            plt.axvline(self.f1, color='red', linestyle='--', alpha=0.7, label=f'截止 {self.f1} Hz')
        elif self.filter_type in ['bandpass', 'bandstop']:
            plt.axvline(self.f1, color='red', linestyle='--', alpha=0.7, label=f'{self.f1} Hz')
            plt.axvline(self.f2, color='red', linestyle='--', alpha=0.7, label=f'{self.f2} Hz')
        elif self.filter_type == 'notch':
            plt.axvline(self.notch_freq, color='red', linestyle='--', alpha=0.7, label=f'陷波 {self.notch_freq} Hz')

        title = f'{self.filter_design.title()} {self.filter_type.title()} 滤波器频率响应'
        if self.filter_type == 'notch':
            title += f'\n陷波频率: {self.notch_freq} Hz, Q: {self.notch_Q}'
        else:
            title += f'\n阶数: {self.order}'

        plt.title(title)
        plt.xlabel('频率 (Hz)')
        plt.ylabel('增益 (dB)')

        # 设置频率范围
        if freq_lim is not None:
            plt.xlim(freq_lim)
        else:
            # 自动设置合适的频率范围
            if self.filter_type == 'lowpass':
                plt.xlim(0.1, self.f1 * 10)
            elif self.filter_type == 'highpass':
                plt.xlim(self.f1 * 0.1, self.fs / 2)
            elif self.filter_type in ['bandpass', 'bandstop']:
                plt.xlim(self.f1 * 0.1, self.f2 * 10)
            elif self.filter_type == 'notch':
                plt.xlim(self.notch_freq * 0.1, self.notch_freq * 10)

        plt.ylim(-80, 5)
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()

        # 相位响应
        plt.subplot(2, 1, 2)
        phase = np.unwrap(np.angle(h))
        plt.semilogx(w, np.degrees(phase))  # 转换为度
        plt.xlabel('频率 (Hz)')
        plt.ylabel('相位 (度)')
        if freq_lim is not None:
            plt.xlim(freq_lim)
        plt.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_info(self):
        """获取滤波器信息"""
        if self.sos is None:
            return "滤波器未设计"

        info = {
            'type': self.filter_type,
            'design': self.filter_design,
            'order': self.order,
            'fs': self.fs,
            # 'sos_shape': self.sos.shape,
            # 'sections': self.sos.shape[0],
            'filtering_method': 'filtfilt (zero-phase)'
        }

        if self.filter_type in ['lowpass', 'highpass']:
            info['cutoff'] = self.f1
        elif self.filter_type in ['bandpass', 'bandstop']:
            info['lowcut'] = self.f1
            info['highcut'] = self.f2
        elif self.filter_type == 'notch':
            info['notch_freq'] = self.notch_freq
            info['Q'] = self.notch_Q

        return info

    def update_parameters(self, **kwargs):
        """更新滤波器参数并重新设计"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"警告: 参数 {key} 不存在")

        # 重新设计滤波器
        self.design_filter()


    # 使用示例和测试函数
    @staticmethod
    def demonstrate_filtfilt_advantage():
        """演示filtfilt双向滤波的优势"""
        fs = 1000
        t = np.linspace(0, 1, fs)

        # 生成测试信号：突发的正弦波
        signal_data = np.zeros_like(t)
        signal_data[200:600] = np.sin(2 * np.pi * 10 * t[200:600])  # 10Hz信号段

        # 创建低通滤波器
        lp_filter = SOSFilter('lowpass', fs, order=4, f1=20, filter_design='butter')

        # 比较两种滤波方式
        filtered_filtfilt = lp_filter.filtfilt(signal_data)  # 双向滤波
        filtered_forward, _ = lp_filter.filter_forward(signal_data)  # 前向滤波

        # 绘制比较结果
        plt.figure(figsize=(12, 10))

        plt.subplot(3, 1, 1)
        plt.plot(t, signal_data, 'b-', linewidth=2)
        plt.title('原始信号 (10Hz突发正弦波)')
        plt.ylabel('幅度')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(t, filtered_forward, 'r-', linewidth=2, label='前向滤波 (有相位延迟)')
        plt.plot(t, signal_data, 'b--', alpha=0.5, label='原始信号')
        plt.title('前向滤波 - 存在相位延迟')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(t, filtered_filtfilt, 'g-', linewidth=2, label='双向滤波 (零相位)')
        plt.plot(t, signal_data, 'b--', alpha=0.5, label='原始信号')
        plt.title('双向滤波 (filtfilt) - 零相位延迟')
        plt.xlabel('时间 (s)')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def example_usage():
        """使用示例"""
        fs = 1000  # 采样率

        print("=== 低通滤波器示例 ===")
        lp_filter = SOSFilter('lowpass', fs, order=4, f1=50, filter_design='butter')
        print(lp_filter.get_info())
        lp_filter.plot_response(freq_lim=(1, 200))

        print("=== 带通滤波器示例 (你的用例) ===")
        bp_filter = SOSFilter('bandpass', fs, order=6, f1=0.4, f2=0.6, filter_design='butter')
        print(bp_filter.get_info())
        bp_filter.plot_response(freq_lim=(0.1, 2))

        print("=== 陷波滤波器示例 ===")
        notch_filter = SOSFilter('notch', fs, notch_freq=50, notch_Q=30)
        print(notch_filter.get_info())

    @staticmethod
    def test_bandpass_low_freq():
        """测试低频带通滤波（你的具体用例）"""
        fs = 1000
        t = np.linspace(0, 10, 10 * fs)

        # 生成包含多种频率成分的信号
        signal_data = (np.sin(2 * np.pi * 0.3 * t) +  # 0.3Hz (在通带外)
                       np.sin(2 * np.pi * 0.5 * t) +  # 0.5Hz (在通带内)
                       0.5 * np.sin(2 * np.pi * 5 * t) +  # 5Hz (在通带外)
                       0.3 * np.sin(2 * np.pi * 50 * t))  # 50Hz (在通带外)

        # 应用0.4-0.6Hz带通滤波
        bp_filter = SOSFilter('bandpass', fs, order=8, f1=0.4, f2=0.6, filter_design='butter')
        filtered = bp_filter.filtfilt(signal_data)

        # 绘制结果
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, signal_data)
        plt.title('原始信号 (包含0.3Hz, 0.5Hz, 5Hz, 50Hz成分)')
        plt.xlabel('时间 (s)')
        plt.ylabel('幅度')
        plt.xlim(0, 5)  # 只看前5秒

        plt.subplot(2, 1, 2)
        plt.plot(t, filtered)
        plt.title('0.4-0.6Hz带通滤波后信号 (双向滤波，零相位)')
        plt.xlabel('时间 (s)')
        plt.ylabel('幅度')
        plt.xlim(0, 5)  # 只看前5秒

        plt.tight_layout()
        plt.show()


def calc_band_psd_by_simpson(f, psd, band: Iterable[Union[float, int]]):
    """
    使用辛普森积分法计算指定频率范围内的功率谱密度积分值,算出来的结果和nfft无关
    Args:
        f: 频率数组，包含功率谱密度对应的频率值
        psd: 功率谱密度数组，与频率数组f相对应的功率谱密度值
        band: 频率范围迭代对象，包含两个元素 [最低频率, 最高频率]
    Returns:
        float: 在指定频率范围内的功率谱密度积分值，表示该频带的总功率
    Notes:
        - 使用辛普森积分法进行数值积分，精度高于梯形法
        - 返回值为该频带的总功率，单位取决于输入psd的单位
        - 频带范围包含边界值（f >= f_min 且 f <= f_max）
    """
    from scipy.integrate import simpson
    f_min, f_max = band
    # 创建频率掩码，选择指定频率范围内的数据点
    mask = (f >= f_min) & (f <= f_max)
    f_band = f[mask]
    psd_band = psd[mask]
    # 使用辛普森法则进行积分（精度更高）
    integrated_power = simpson(y=psd_band, x=f_band)
    return integrated_power

def calc_psd(lfp_data: np.ndarray|list,
                 fs: float = 1000,
                 nperseg: int = 1024,
                 scaling: str = 'density',) -> tuple:
    """
    计算LFP数据的功率谱密度，注意welch默认返回的单边谱就已经是乘以2的结果了，不用再乘了

    参数:
    lfp_data: 输入的LFP数据，应为1维数组
    fs: 采样频率 (Hz)，默认1000Hz
    nperseg: Welch方法中每个段的长度，默认1024
    scaling: Welch方法返回的是psd还是功率，注意psd/功率是幅值的平方

    返回:
    tuple: (频率数组, 功率谱密度数组)
    """
    from scipy import signal
    # 检查输入数据维度
    if len(lfp_data.shape) > 1:
        raise ValueError("输入数据不为1维，请提供单通道LFP数据")

    # 计算功率谱密度
    f, psd = signal.welch(lfp_data,
                          fs=fs,
                          nperseg=nperseg,
                          scaling=scaling)
    return f, psd


def calc_heatmap_PAC(lfp_value: np.ndarray,
                     ipdac: list or np.ndarray,
                     freq_amp: list or np.ndarray,
                     freq_ang: list or np.ndarray,
                     fs: float,
                     time_series: list,
                     df_name: str,
                     figure_name: str,
                     pad_time: int = 150,
                     n_jobs_filter: int = -1,
                     n_jobs_pac: int = -1,
                     ) -> None:
    """
    计算事件相关的相位-幅度耦合(PAC)并生成热图

    Args:
        lfp_value (np.ndarray): 一维局部场电位(LFP)信号数据，形状为(n_samples,)
        ipdac: 用于计算相幅耦合的tensorpac库的ipdac
        freq_amp (list or np.ndarray): 幅度频率范围，用于高频成分分析，格式为[start, stop, n_steps]
        freq_ang (list or np.ndarray): 相位频率范围，用于低频相位分析，格式为[start, stop, n_steps]
        fs (float): 采样频率(Hz)
        time_series: 事件发生时间点序列，每个元素代表事件发生的时间点（采样点索引）
        df_name (str): 保存结果的Excel文件名
        figure_name (str): 保存PAC热图的图像文件名
        pad_time (int): 填充时间点数，用于在事件时间点前后截取数据

    Returns:
        None: 结果保存为Excel文件和图像文件

    Example:
        # >>> lfp_data = np.random.randn(10000)
        # >>> freq_amp = [60, 120, 10,5]  # 60-120Hz，10的窗长，5的步长
        # >>> freq_ang = [4, 12, 8,4]     # 4-12Hz，8的窗长，4的步长
        # >>> event_times = [1000, 2000, 3000]  # 事件发生的时间点
        # >>> calc_heatmap_PAC(lfp_data, freq_amp, freq_ang, 1000, event_times,
        # ...                 'results.xlsx', 'pac_heatmap.png', pad_time=150)
    """
    import matplotlib.pyplot as plt
    from tensorpac import Pac
    import matplotlib
    import numpy as np
    import pandas as pd
    import tempfile
    import os

    def saveas_tempfile(large_array: np.ndarray) -> str:
        """
        将大数组保存到临时文件，避免内存占用过高
        Args:
            large_array: 要保存的数组
        Returns:
            str: 临时文件路径
        """
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            np.savez(tmp_path, large_array=large_array)
            return tmp_path

    def load_tempfile(tmp_path: str) -> np.ndarray:
        """
        从临时文件加载数组
        Args:
            tmp_path: 临时文件路径
        Returns:
            np.ndarray: 加载的数组
        """
        with np.load(tmp_path) as data:
            loaded_array = data['large_array']
        return loaded_array

    def calc_slicing_amps_or_phases(p: Pac,
                                    lfp_value: np.ndarray,
                                    fs: float,
                                    event_times: list,
                                    pad_time: int,
                                    ftype: str,
                                    n_jobs: int) -> np.ndarray:
        """
        提取事件时间点附近的相位或幅值序列
        Args:
            p: Pac对象，用于信号滤波
            lfp_value: LFP信号数据
            fs: 采样频率
            event_times: 事件发生时间点列表
            pad_time: 事件前后填充的时间点数
            ftype: 滤波类型，'phase'或'amplitude'
            n_jobs: 并行处理线程数
        Returns:
            np.ndarray: 形状为(频率数量, 事件数量, 2*pad_time)的数组
        """
        # 对LFP信号进行滤波，提取相位或幅度信息
        # 使用Hilbert变换方法
        filtered_data = p.filter(fs, lfp_value, ftype=ftype, n_jobs=n_jobs)
        # 初始化存储事件数据的数组
        # 形状: (频率数量, 事件数量, 时间点数)
        event_data = np.zeros(shape=(filtered_data.shape[0], len(event_times), 2 * pad_time))
        # 提取每个事件周围的数据段
        for idx, event_time in enumerate(event_times):
            start_time = event_time - pad_time  # 起始时间点（事件前pad_time个点）
            end_time = event_time + pad_time  # 结束时间点（事件后pad_time个点）
            # 检查边界有效性
            if start_time >= 0 and end_time <= filtered_data.shape[2]:
                # 提取数据段，squeeze移除单维度
                event_data[:, idx, :] = filtered_data[:, :, start_time:end_time].squeeze(axis=1)
            else:
                # 如果边界越界，填充NaN
                event_data[:, idx, :] = np.nan
                print(f"警告: 事件 {idx} 时间点 {event_time} 超出信号边界")
        return event_data

    def figure_PAC(mi_array: np.ndarray, p: Pac, figure_name: str) -> None:
        """
        绘制并保存相位-幅度耦合(PAC)热图
        Args:
            mi_array (np.ndarray): 调制指数矩阵，形状为(n_amp_freq, n_pha_freq)
            p (Pac): 已初始化的Pac对象，包含频率信息
            figure_name (str): 保存图像的文件名
        """
        # 绘制comodulogram热图，显示调制指数
        ax = p.comodulogram(mi_array,
                            # cmap='plasma',  # 可选：使用其他颜色映射
                            fz_labels=6, fz_title=12, fz_cblabel=6, title='Modulation Index (MI)')
        # 查找颜色条对象
        cbar = None
        for im in ax.get_images():
            if im.colorbar is not None:  # 检查是否有颜色条
                cbar = im.colorbar
                break
        # 设置科学计数法显示，适用于调制指数通常较小的情况
        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)  # 强制使用科学计数法
        formatter.set_powerlimits((2, 10))  # 当数值在10^2到10^10范围时使用科学计数法
        # 调整数量级文本的位置和字体大小
        if cbar is not None:
            offset_text = cbar.ax.yaxis.get_offset_text()
            offset_text.set_x(1.5)  # 向右移动数量级文本，避免与刻度标签重叠
            offset_text.set_fontsize(6)
            # 应用科学计数法到颜色条
            cbar.ax.yaxis.set_major_formatter(formatter)
            # 调整颜色条刻度字体大小
            cbar.ax.tick_params(labelsize=6)
        # 调整坐标轴刻度字体大小
        ax.tick_params(axis='both', labelsize=6)  # X、Y轴刻度
        # 保存图像，格式由文件名后缀决定(如.png, .pdf, .svg等)
        p.savefig(figure_name, dpi=300)  # 设置高分辨率输出
        plt.close()  # 关闭图形，释放内存

    # ==================== 主函数逻辑开始 ===================

    # 步骤1: 筛选有效的事件时间点（确保不超出信号边界）
    valid_event_times = [t for t in time_series
                         if (t >= pad_time) & (t + pad_time <= len(lfp_value) - 1)]
    # 检查是否有有效的事件
    if len(valid_event_times) == 0:
        raise ValueError("未找到有效的事件时间点，请检查pad_time参数或时间序列数据")
    if len(valid_event_times) < len(time_series):
        print(f"警告: {len(time_series) - len(valid_event_times)} 个事件因边界问题被排除")
    # 步骤2: 初始化Pac对象，使用特定的PAC计算方法
    # idpac=(2, 0, 0) 表示使用Mean Vector Length方法
    p = Pac(idpac=ipdac, f_pha=freq_ang, f_amp=freq_amp, dcomplex='hilbert')
    # 获取实际的频率范围（可能因滤波过程略有调整）
    f_amp = p.f_amp  # 幅度频率范围，形状为(n_amp_bands, 2)
    f_pha = p.f_pha  # 相位频率范围，形状为(n_pha_bands, 2)
    # 步骤3: 提取相位序列（低频成分）
    print("提取相位序列...")
    phases_events = calc_slicing_amps_or_phases(p, lfp_value, fs, valid_event_times,
                                                pad_time, ftype='phase', n_jobs=n_jobs_filter)

    # 数组过大，保存为临时文件以释放内存
    tmp_path_phases = saveas_tempfile(phases_events)
    del phases_events  # 立即删除原数组释放内存

    # 步骤4: 提取幅度序列（高频成分）
    print("提取幅度序列...")
    amps_events = calc_slicing_amps_or_phases(p, lfp_value, fs, valid_event_times,
                                              pad_time, ftype='amplitude', n_jobs=n_jobs_filter)

    # 重新加载相位数据
    phases_events = load_tempfile(tmp_path_phases)
    # 使用完后立即删除临时文件
    os.unlink(tmp_path_phases)

    # 步骤5: 计算调制指数(MI) - 衡量相位-幅度耦合强度
    # 返回的mi_array形状为(n_amp_freq, n_pha_freq, n_timepoints)
    mi_array = p.fit(phases_events, amps_events, n_jobs=n_jobs_pac)

    # 释放大数组内存
    del phases_events, amps_events

    # 步骤6: 沿时间维度取平均值，得到最终的MI矩阵
    # 形状变为: (n_amp_freq, n_pha_freq, n_events)
    mi_array = np.nanmean(mi_array, axis=-1)

    # 如果还有事件维度，继续平均（取决于tensorpac版本）
    if mi_array.ndim > 2:
        mi_array = np.nanmean(mi_array, axis=-1)

    # 步骤7: 绘制并保存PAC热图
    figure_PAC(mi_array, p, figure_name)

    # 步骤8: 创建DataFrame保存结果数据
    # 索引为幅度频率的中心值，列为相位频率的中心值
    amp_freq_centers = np.mean(f_amp, axis=-1)  # 计算每个幅度频带的中心频率
    pha_freq_centers = np.mean(f_pha, axis=-1)  # 计算每个相位频带的中心频率

    df_results = pd.DataFrame(mi_array,
                              index=amp_freq_centers,
                              columns=pha_freq_centers)

    # 设置DataFrame的行列名称，便于识别
    df_results.index.name = 'Amplitude Frequency (Hz)'
    df_results.columns.name = 'Phase Frequency (Hz)'

    # 步骤9: 将结果保存到Excel文件
    with pd.ExcelWriter(df_name, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='pac_results')

    print(f" 分析完成 - 结果保存至: {df_name} 和 {figure_name}")

    # 返回结果DataFrame（可选）
    return df_results



class RippleDetector:
    # noinspection PyUnresolvedReferences
    def __init__(self):
        """
                初始化Ripple检测器参数

                参数说明：

                基本参数：
                    fs: 采样率，1000Hz表示每秒采集1000个数据点

                带通滤波参数：
                    bandpass_freq: 带通滤波频率范围[100, 250]Hz，用于提取ripple振荡信号
                    N_bandpass: 滤波器阶数，阶数越高滤波越陡峭

                陷波滤波参数：
                    notch_f0: 陷波滤波中心频率50Hz，用于去除工频干扰
                    notch_bandwidth: 陷波滤波带宽1Hz，带宽越窄去除的频率范围越小

                高斯包络参数：
                    gauss_sigma: 高斯函数标准差5.33（采样点），控制包络平滑程度
                    gauss_truncate: 高斯函数截断范围4.0，表示使用±4倍sigma范围内的数据

                Ripple检测参数（所有时间参数均以采样点数为单位，1000采样点=1秒）：
                    half_window_len: 半窗口长度3000采样点（3秒），用于计算滑动阈值
                    high_fold: 高阈值倍数5，阈值=均值+5*标准差，用于检测ripple峰值
                    low_fold: 低阈值倍数2，阈值=均值+2*标准差，用于确定ripple起止点
                    continuous_time: 连续时间7采样点（7ms），例如某个高于high_fold的序列中有5个连续点的值小于high_fold，
                        这样不会被视为2个ripple，只有当小于high_fold的连续点数大于7个，这段高于high_fold的序列才会被识别为2个ripple。
                    epoch_high_time: 高阈值持续时间10采样点（10ms），ripple必须包含至少10个采样点超过高阈值
                    ripple_concat_time: ripple合并间隔50采样点（50ms），间隔小于此时间的ripple合并
                    ripple_longest_time: ripple最长持续时间450采样点（450ms），超过此时间的不认为是ripple
                    ripple_shortest_time: ripple最短持续时间50采样点（50ms），短于此时间的不认为是ripple
                    wrong_dots_threshold: 异常点阈值20，如果某个ripple的时间内有任何一个点的z-score大于该值，则删掉这个ripple

                绘图参数：
                    is_plot: 是否生成绘图，True
                    figure_width: 图形宽度参数200，具体用途请确认（可能是像素或缩放比例）
                    html_path: HTML文件保存的文件夹的路径
                    figure_name: 图形名称'阈值图'
                """

        global np,pd,signal,gaussian_filter,os,go,gc,Tuple,Path,yyl
        import numpy as np
        import pandas as pd
        from scipy import signal
        from scipy.ndimage import gaussian_filter
        import os
        import plotly.graph_objects as go
        import gc
        from typing import Tuple
        from pathlib import Path
        import yyl_utils as yyl


        self.fs=1000

        # 带通滤波参数
        self.bandpass_freq=[100,250]
        self.N_bandpass=4

        # notch滤波参数
        self.notch_f0=50
        self.notch_bandwidth=1

        # 计算高斯包络参数
        self.gauss_sigma=5.33
        self.gauss_truncate=4.0

        # Ripple检测参数, 其中涉及数值均为采样点数
        self.half_window_len = 3000      #进行滑动计算均值和SD的窗长的一半
        self.high_fold = 5      #高阈值检测限的SD倍数
        self.low_fold = 2       #低阈值检测限的SD倍数
        self.continuous_time = 7       #高频间间距小于continuous_time个采样点的ripple会被合并
        self.epoch_high_time = 10      #高阈值部分至少要持续epoch_high_time个采样点
        self.ripple_concat_time = 50   #低频间间距小于ripple_concat_time个采样点的ripple会被合并
        self.ripple_longest_time =450      #ripple的最长持续时间不多于ripple_longest_time个采样点
        self.ripple_shortest_time = 50      #ripple的最短持续时间不少于ripple_longest_time个采样点
        self.soft_wrong_dots_threshold = 20 #阈值检测，该阈值是自适应的阈值，单位是(倍SD)
        self.hard_wrong_dots_threshold = 3  #阈值检测，该阈值是限定的阈值，单位跟随lfp的单位

        # 画html图参数
        self.is_plot = False
        self.figure_width = 200
        self.figure_num_waveforms = 500
        self.html_path = Path(r"")
        self.figure_name='阈值图'

    def mybandpassfilter(self, lfp):
        """
        带通滤波函数，对lfp进行带通滤波
        :param lfp: 输入信号数据
        :return: 滤波后的信号
        """
        nyquist = 0.5 * self.fs
        Wn = (self.bandpass_freq[0] / nyquist, self.bandpass_freq[1] / nyquist)
        # 使用SOS形式设计滤波器
        sos = signal.butter(self.N_bandpass, Wn, btype='bandpass', output='sos')
        # 使用sosfiltfilt进行零相位滤波
        filtered_lfp = signal.sosfiltfilt(sos, lfp)
        return filtered_lfp

    def my_notchfilter(self,lfp):
        """
            应用 Notch 滤波器去除特定频率的干扰信号。
            参数:
            - data: 输入信号 (1D 或 2D 数组)
            - fs: 采样率 (Hz)
            - f0: 滤波器中心频率 (默认 50Hz)
            - bandwidth: 滤波器带宽 (默认 3Hz)
            返回:
            - filt_data: 滤波后的信号
            """
        w0 = self.notch_f0 / (self.fs / 2)  # 归一化中心频率（Nyquist 频率归一化）
        Q = self.notch_f0 / self.notch_bandwidth
        b, a = signal.iirnotch(w0=w0, Q=Q)
        filt_lfp = signal.filtfilt(b, a, lfp)
        return filt_lfp

    def calc_gauss_envelope(self,raw_data:np.ndarray)->np.ndarray:
        """
        计算数据的高斯包络,通过希尔伯特变换获取包络，然后进行高斯平滑
        :param raw_data:
        :return: 返回高斯包络
        """
        data_envelope = np.abs(signal.hilbert(raw_data))
        data_gauss_envelope = gaussian_filter(data_envelope,
              sigma=self.gauss_sigma, truncate=self.gauss_truncate)
        return data_gauss_envelope

    # 得到包络线的阈值序列
    def get_threshold_sd(self, envelope_values,) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        功能：计算高低阈值和Z-score序列
        计算公式：
            threshold_high_sd = 局部均值 + high_fold × 局部标准差
            threshold_low_sd = 局部均值 + low_fold × 局部标准差
            z_score_line = (原始值 - 局部均值) / 局部标准差
        用途：生成自适应的动态阈值
        :param envelope_values:
        :param half_window_len:
        :param high_fold:
        :param low_fold:
        :return:
        返回的是高阈值、低阈值、Z线的值，用于后续作图
        """
        # 获取lfp的均值和std序列
        def get_mean_and_std(envelope_values, half_window_len) -> Tuple[np.ndarray, np.ndarray]:
            """
            功能：计算包络线的滑动均值和标准差
            处理逻辑：
                使用pandas的rolling窗口计算,采用滑动窗的方法计算该段的均值和标准差
                窗口大小：2*half_window_len+1（中心对称窗口）
            输出：每个点的局部均值和标准差
            :param envelope_values:
            :param half_window_len:
            :return:返回与输入序列等长的滑动平均后的序列，以及每个点上的std值
            """
            envelope_values_series = pd.Series(envelope_values)
            window_mean = envelope_values_series.rolling(2 * half_window_len + 1, min_periods=1,
                                                         center=True).mean()
            window_std = envelope_values_series.rolling(2 * half_window_len + 1, min_periods=1,
                                                        center=True).std()
            window_mean = window_mean.to_numpy()
            window_std = window_std.to_numpy()
            return window_mean, window_std

        window_mean, window_std = get_mean_and_std(envelope_values, self.half_window_len)
        threshold_high_sd = window_mean + self.high_fold * window_std
        threshold_low_sd = window_mean + self.low_fold * window_std
        z_score_line = (envelope_values - window_mean) / (window_std + 1e-12)
        return threshold_high_sd, threshold_low_sd, z_score_line

    def get_ripple_time(self,envelope_values:np.ndarray, notched_lfp) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            # 将超过阈值的index进行融合，得到每个epoch的开始时间和结束时间。如果两个超过阈值时间小于continuous_time则视为一个epoch。
            # 返回值是一个np.ndarray，矩阵中元素是n行2列，每行的两个元素由每个ripple的起止index组成。
            # 输入是包络线,输出是符合要求的ripple的开始、结束时间。
            # 要求：1.小于ripple_concat_time的ripple合并
            # 2.小于continuous_time的阈下时间跳过
            # 3.要有高于high_fold的部分，起止时间是高于low_fold的部分
            # 4.高于high_fold的时间要多于epoch_high_time
            # 5.最终的ripple时间要大于ripple_shortest_time，小于ripple_longest_time
            # 6.坏点检测：①如果ripple的时间内有任何一个点的z-score大于soft_wrong_dots_threshold，则删掉这些ripple
            #           ②如果ripple的时间内有任何一个点的notched_lfp大于hard_wrong_dots_threshold，则删掉这些ripple
            # 此处的所有涉及到的时间都是以采样点数计算，而不是现实时间。
        :param envelope_values: 根据包络序列计算ripple时间,
        notched_lfp：notch后的lfp，用于硬阈值去噪
        :return: epoches_ripple：ripple的起止时间,
                threshold_high_sd：高阈值序列(用于画图), threshold_low_sd：低阈值序列(用于画图), z_score_line：Z线(用于画图)
        """

        # 输入是包络线，输出是超过阈值的epoch的起止时间
        def get_over_threshold_sequence(envelope_values, threshold_high_sd, threshold_low_sd, continuous_time):
            """
            功能：检测超过高低阈值的时间段
            处理流程：
                获取高低阈值序列
                分别找出超过高阈值和低阈值的时间点
                使用get_epoch将时间点合并为连续时段
            输出：高阈值时段、低阈值时段、阈值序列、Z-score
            :param envelope_values:
            :param half_window_len:
            :param high_fold:
            :param low_fold:
            :param continuous_time:
            :return:
            epoch_high_sd:一个list，其中元素为2个元素构成的tuple，tuple中的值为超过高阈值的始末点
            epoch_low_sd：类似epoch_high_sd
            threshold_high_sd：与输入序列等长的高阈值线的值,用于后续作图
            threshold_low_sd：类似threshold_high_sd,
            z_score_line：与输入序列等长的Z线的值，用于后续作图
            """

            def get_epoch(overthreshold_time_sequence: list, continuous_time: int) -> list[Tuple[int, int]]:
                """
                功能：将离散的超过阈值的时间点合并成连续的时间段
                处理逻辑：
                如果相邻时间点间隔 ≤ continuous_time，则合并为同一个时段
                :param overthreshold_time_sequence:超过阈值的时间点序列 [t1, t2, t3, ...]
                :param continuous_time:
                :return:时段列表 [(start1, end1), (start2, end2), ...]
                """
                epoch_time = []
                n1 = 0
                while n1 < len(overthreshold_time_sequence):
                    start_epoch_time = overthreshold_time_sequence[n1]
                    while n1 < len(overthreshold_time_sequence) - 1 and (
                            overthreshold_time_sequence[n1 + 1] - overthreshold_time_sequence[n1] <= continuous_time):
                        n1 += 1
                    end_epoch_time = overthreshold_time_sequence[n1]
                    epoch_time.append((start_epoch_time, end_epoch_time))
                    n1 += 1
                return epoch_time

            idx_high_sd = []
            idx_low_sd = []
            for n1, value in enumerate(envelope_values):
                if np.abs(value) > threshold_high_sd[n1]:
                    idx_high_sd.append(n1)
                if np.abs(value) > threshold_low_sd[n1]:
                    idx_low_sd.append(n1)
            epoch_high_sd = get_epoch(idx_high_sd, continuous_time)
            epoch_low_sd = get_epoch(idx_low_sd, continuous_time)
            return epoch_high_sd, epoch_low_sd

        def get_low_high_overlap(epoch_low, epoch_high):
            """筛选出epoch_low中包含至少一个epoch_high的epoch"""
            # 先检查列表是否为空
            if len(epoch_low) == 0 or len(epoch_high) == 0:
                return []

            # 再转换为NumPy数组
            low_arr = np.array(epoch_low)
            high_arr = np.array(epoch_high)

            # 使用广播加速
            overlaps = []
            for low_start, low_end in low_arr:
                # 向量化检查：high是否在low内
                mask = (high_arr[:, 0] >= low_start) & (high_arr[:, 1] <= low_end)
                if np.any(mask):
                    overlaps.append((low_start, low_end))

            return overlaps

        def concat_continuous_ripple(raw_epoch,ripple_concat_time):
            """
            将ripple间间隔小于ripple_concat_time的合并
            :param raw_epoch:
            :param ripple_concat_time:
            :return: 返回合并后的epoch
            """
            concated_epoch = []
            n1 = 0
            while n1 < len(raw_epoch) - 1:
                if raw_epoch[n1 + 1][0] - raw_epoch[n1][1] <= ripple_concat_time:
                    concated_epoch.append((raw_epoch[n1][0], raw_epoch[n1 + 1][1]))
                    n1 += 2
                    continue
                if raw_epoch[n1 + 1][0] - raw_epoch[n1][1] > ripple_concat_time:
                    concated_epoch.append(raw_epoch[n1])
                    n1 += 1
            if n1 == len(raw_epoch) - 1:  # 当n1是最后一个元素时
                concated_epoch.append(raw_epoch[n1])
            return concated_epoch

        def screen_appropriate_time_ripple(raw_epoch,ripple_shortest_time,ripple_longest_time):
            """
            将持续时间大于ripple_longest_time的ripple去掉
            :return:
            """
            screened_epoch = []
            for item in raw_epoch:
                if (item[1] - item[0] <= ripple_longest_time) & (item[1] - item[0] >= ripple_shortest_time):
                    screened_epoch.append(item)
            return screened_epoch

        def delete_wrong_dots(ripple_series, z_score_line, soft_wrong_dots_threshold):
            '''判断z_score_line中在时间范围ripple_series内是否有值大于soft_wrong_dots_threshold的点，并删除这些点'''
            ripple_series = [epoch for epoch in ripple_series
                             if not np.any(z_score_line[epoch[0]:epoch[1]] >= soft_wrong_dots_threshold)]
            return ripple_series

        threshold_high_sd, threshold_low_sd, z_score_line = (
            self.get_threshold_sd(envelope_values))
        epoch_high_sd, epoch_low_sd = get_over_threshold_sequence(
            envelope_values, threshold_high_sd, threshold_low_sd, self.continuous_time)
        # 筛选ripple的超过高阈值的时间大于self.epoch_high_time的ripple事件
        epoch_high_sd_2 = [epoch for epoch in epoch_high_sd if epoch[1] - epoch[0] >= self.epoch_high_time]
        # 获取包含epoch_high的epooch_low
        epoch_overlap_low = get_low_high_overlap(np.array(epoch_low_sd), np.array(epoch_high_sd_2))
        # 融合相距过近的ripple事件
        epoch_concated=concat_continuous_ripple(epoch_overlap_low,self.ripple_concat_time)
        # 筛选出时长在self.ripple_shortest_time和self.ripple_longest_time范围内的ripple
        epoch_screened=screen_appropriate_time_ripple(epoch_concated,self.ripple_shortest_time,self.ripple_longest_time)
        # 去除噪音过大的点
        if self.soft_wrong_dots_threshold :
            # 软阈值去除超过self.soft_wrong_dots_threshold*STD的点
            epoch_screened = delete_wrong_dots(epoch_screened, z_score_line, self.soft_wrong_dots_threshold)
        if self.hard_wrong_dots_threshold :
            # 去除notch后lfp中的值超过self.hard_wrong_dots_threshold的点
            epoch_screened = delete_wrong_dots(epoch_screened, notched_lfp, self.hard_wrong_dots_threshold)
        epoches_ripple = epoch_screened
        # 对ripple事件按时间排序
        epoches_ripple = sorted(epoches_ripple, key=lambda x: x[0])
        epoches_ripple = np.array(epoches_ripple)
        print(f'检测结束，共检测到{len(epoches_ripple)}个ripple')
        return epoches_ripple, threshold_high_sd, threshold_low_sd, z_score_line

    def plot_html(self, lfp, envelope, figure_width, num_waveforms, threshold_high_sd, threshold_low_sd,
                  htmlTotalPath:Path, file_name:str,
                  epoches_ripple):
        """
        生成交互式的脑电波纹(Ripple)检测结果可视化HTML图表

        参数:
            lfp: 原始局部场电位(LFP)信号数据
            envelope: 包络线信号数据
            figure_width: 图形宽度参数，用于数据填充
            threshold_high_sd: 高阈值标准差值
            threshold_low_sd: 低阈值标准差值
            htmlTotalPath: HTML文件保存路径
            file_name: 输出文件的基础名称
            epoches_ripple: 检测到的波纹事件时段列表，每个元素为[start, end]格式
        """

        # 画交互式图
        def create_interactive_ripple_plot_with_lines(ripple_line, idx, envelope_line,
                                                      threshold_high_sd_line, threshold_low_sd_line, output_file):
            """
            创建并保存交互式的Ripple可视化图表

            参数:
                ripple_line: 包含波纹数据的数组
                idx: 竖线位置索引数组，标记每个波纹事件的结束位置
                envelope_line: 包络线数据
                threshold_high_sd_line: 高阈值线数据
                threshold_low_sd_line: 低阈值线数据
                output_file: 输出的HTML文件名
            """
            # 创建主波纹信号轨迹
            trace1 = go.Scatter(
                x=np.arange(len(ripple_line)),
                y=ripple_line,
                mode='lines',
                line=dict(color='blue'),
                name="ripple_line"
            )

            # 创建详细视图的波纹信号轨迹（与trace1相同，用于双视图布局）
            trace2 = go.Scatter(
                x=np.arange(len(ripple_line)),
                y=ripple_line,
                mode='lines',
                line=dict(color='blue'),
                name="ripple_line"
            )

            # 创建包络线轨迹
            trace3 = go.Scatter(
                x=np.arange(len(envelope_line)),
                y=envelope_line,
                mode='lines',
                line=dict(color='orange'),
                name="envelope_line"
            )
            # 创建高阈值线轨迹
            trace4 = go.Scatter(
                x=np.arange(len(threshold_high_sd_line)),
                y=threshold_high_sd_line,
                mode='lines',
                line=dict(color='red'),
                name="threshold_high_sd_line"
            )

            # 创建低阈值线轨迹
            trace5 = go.Scatter(
                x=np.arange(len(threshold_low_sd_line)),
                y=threshold_low_sd_line,
                mode='lines',
                line=dict(color='red'),
                name="threshold_low_sd_line"
            )

            # 添加竖线标记每个波纹事件的结束位置
            vlines = []
            for x_pos in idx:
                vlines.append(go.Scatter(
                    x=[x_pos, x_pos],
                    y=[min(ripple_line), max(ripple_line)],
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False,
                ))

            # 创建图表布局
            layout = go.Layout(
                title="Ripple Line Visualization",
                showlegend=True,
                xaxis=dict(
                    title="Index",
                    rangeslider=dict(visible=True),  # 显示范围滑块
                    domain=[0, 1]  # 缩略图占据整个x轴
                ),
                yaxis=dict(
                    title="Value",
                    autorange=True,  # 纵轴自动缩放
                    fixedrange=False  # 确保纵轴允许缩放
                ),
                xaxis2=dict(
                    title="Index",
                    range=[0, 1000],  # 初始显示的x轴范围
                    domain=[0, 1],  # 下图占据整个x轴
                ),
                yaxis2=dict(
                    title="Value",
                    autorange=True,  # 纵轴自动缩放
                    fixedrange=False  # 确保纵轴允许缩放
                ),
                plot_bgcolor="white",
                dragmode="pan",  # 启用平移模式
                hovermode="closest"
            )

            # 创建包含所有轨迹的图表
            fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5] + vlines, layout=layout)

            # 配置交互功能
            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),  # 可调范围滑块
                ),
                xaxis2=dict(
                    rangeslider=dict(visible=True),  # 下面图表也显示范围滑块
                )
            )

            # 将图表保存为HTML文件
            yyl.make_sure_folder_exist(Path(output_file).parent)
            fig.write_html(output_file)
            print(f"HTML file has been saved successfully as {output_file}!")
            # 确保释放Plotly对象
            if 'fig' in locals():
                del fig
            gc.collect()

        # 初始化数据存储列表
        ripple_line_row = []  # 存储波纹信号数据
        axv_idx_row = []  # 存储竖线位置索引
        evnelope_line_row = []  # 存储包络线数据
        threshold_high_sd_line_row = []  # 存储高阈值线数据
        threshold_low_sd_line_row = []  # 存储低阈值线数据

        # 对数据进行填充，确保波纹事件前后有足够的显示范围
        lfp_value_padded = np.pad(lfp, (figure_width, figure_width), mode='constant', constant_values=0)
        envelope_padded = np.pad(envelope, (figure_width, figure_width), mode='constant', constant_values=0)
        threshold_high_sd_padded = np.pad(
            threshold_high_sd, (figure_width, figure_width), mode='constant', constant_values=0)
        threshold_low_sd_padded = np.pad(
            threshold_low_sd, (figure_width, figure_width), mode='constant', constant_values=0)

        # 设置初始输出文件名
        output_file_name = os.path.join(htmlTotalPath, file_name + '_0.html')

        # 遍历所有检测到的波纹事件
        for n1, item in enumerate(epoches_ripple):
            # 提取当前波纹事件及其前后扩展区域的数据
            epoch = lfp_value_padded[item[0]:item[1] + 2 * figure_width]
            envelope_epoch = envelope_padded[item[0]:item[1] + 2 * figure_width]
            threshold_high_sd_epoch = threshold_high_sd_padded[item[0]:item[1] + 2 * figure_width]
            threshold_low_sd_epoch = threshold_low_sd_padded[item[0]:item[1] + 2 * figure_width]

            # 如果不是最后一个事件
            if (n1 + 1) != len(epoches_ripple):
                # 每处理num_waveforms个事件就生成一个HTML文件
                if (n1 + 1) % num_waveforms == 0:
                    create_interactive_ripple_plot_with_lines(
                        ripple_line_row, axv_idx_row, evnelope_line_row,
                        threshold_high_sd_line_row, threshold_low_sd_line_row,
                        output_file=output_file_name)
                    print(f"作图中，正进行第{(n1 + 1)}/{len(epoches_ripple)}个")

                    # 重置数据列表，开始新的一批数据处理
                    ripple_line_row = []
                    axv_idx_row = []
                    evnelope_line_row = []
                    threshold_high_sd_line_row = []
                    threshold_low_sd_line_row = []

                    # 添加当前事件数据到新列表
                    ripple_line_row.extend(epoch)
                    evnelope_line_row.extend(envelope_epoch)
                    threshold_high_sd_line_row.extend(threshold_high_sd_epoch)
                    threshold_low_sd_line_row.extend(threshold_low_sd_epoch)
                    axv_idx_row.append(len(ripple_line_row) - 1)  # 记录当前事件的结束位置

                    # 更新输出文件名
                    output_file_name = htmlTotalPath/f'{file_name}_{(n1 + 1) // num_waveforms}.html'

                # 如果未达到num_waveforms个事件，继续累积数据
                if (n1 + 1) % num_waveforms != 0:
                    ripple_line_row.extend(epoch)
                    evnelope_line_row.extend(envelope_epoch)
                    threshold_high_sd_line_row.extend(threshold_high_sd_epoch)
                    threshold_low_sd_line_row.extend(threshold_low_sd_epoch)
                    axv_idx_row.append(len(ripple_line_row) - 1)  # 记录当前事件的结束位置

            # 处理最后一个事件
            if (n1 + 1) == len(epoches_ripple):
                create_interactive_ripple_plot_with_lines(
                    ripple_line_row, axv_idx_row, evnelope_line_row,
                    threshold_high_sd_line_row, threshold_low_sd_line_row,
                    output_file=output_file_name)

    def total_pipeline(self,lfp:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        输入的是用于计算的lfp，输出的是epoches_ripple, envelope_lfp, threshold_high_sd, threshold_low_sd, z_score_line,分别是
        ripple发作时间(是一个n行2列的矩阵，矩阵的每行是[开始时间，结束时间])，lfp的包络线，
        用于作图的整个时段的高阈值线、低阈值线，滤波后lfp的幅值包络的z线
        :param lfp:
        :return:
        """
        notched_lfp=self.my_notchfilter(lfp)
        bandpassed_lfp=self.mybandpassfilter(notched_lfp)
        envelope_lfp=self.calc_gauss_envelope(bandpassed_lfp)
        epoches_ripple, threshold_high_sd, threshold_low_sd, z_score_line=self.get_ripple_time(envelope_lfp, notched_lfp)
        # 画html图
        if (self.is_plot==True) & (len(epoches_ripple) != 0):
            self.plot_html(notched_lfp,envelope_lfp,self.figure_width,self.figure_num_waveforms, threshold_high_sd,
                           threshold_low_sd,self.html_path,self.figure_name,epoches_ripple)
        return epoches_ripple, envelope_lfp, bandpassed_lfp, threshold_high_sd, threshold_low_sd, z_score_line,