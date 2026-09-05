"""
WaveClus 3.0 的 Python 封装。

文件结构：
1. WaveClusBatchSorter：管理一个 MATLAB Engine 会话；
2. extract_spikes：批量调用 MATLAB Get_spikes；
3. cluster_waveforms：批量调用 MATLAB Do_clustering，并生成 HDF5；
4. save_recording_mat / save_waveforms_mat：把 NumPy 数组保存为 WaveClus 可读的 MAT；
5. describe_settings：打印 WaveClus 可设置参数、默认值和说明。

使用前需要在当前 Python 环境中安装 MATLAB Engine API，例如：
    cd "<MATLAB安装目录>\\extern\\engines\\python"
    python -m pip install .

本文件不在模块导入阶段导入 numpy、scipy 或 matlab.engine，相关依赖均采用懒加载。
"""


class WaveClusBatchSorter:
    """通过一个 MATLAB Engine 会话批量运行 WaveClus。"""

    def __init__(self, root_dir, matlab_file=None, auto_start=True):
        """初始化封装。

        Parameters
        ----------
        root_dir : str or os.PathLike
            根目录。根目录下应包含 wave_clus-master 文件夹和本封装的 MATLAB 文件。
        matlab_file : str or os.PathLike, optional
            MATLAB 封装文件路径；默认是 root_dir/waveclus_wrapper.m。
        auto_start : bool
            是否在初始化时启动 MATLAB Engine。False 可在第一次调用时再启动。
        """
        import os

        self.root_dir = os.path.abspath(os.fspath(root_dir))
        if matlab_file is None:
            matlab_file = os.path.join(self.root_dir, "waveclus_wrapper.m")
        self.matlab_file = os.path.abspath(os.fspath(matlab_file))
        if not os.path.isfile(self.matlab_file):
            raise FileNotFoundError("找不到 MATLAB 封装文件: " + self.matlab_file)

        self.waveclus_dir = os.path.join(self.root_dir, "wave_clus-master")
        if not os.path.isdir(self.waveclus_dir):
            raise FileNotFoundError("根目录下找不到 wave_clus-master: " + self.waveclus_dir)

        self._engine = None
        if auto_start:
            self._get_engine()

    def _get_engine(self):
        """启动并缓存 MATLAB Engine，保证整个批处理只启动一次 MATLAB。"""
        import os

        if self._engine is None:
            try:
                import matlab.engine
            except ImportError as exc:
                raise ImportError(
                    "当前 Python 环境没有 MATLAB Engine API，请先安装 matlabengine。"
                ) from exc

            self._engine = matlab.engine.start_matlab()
            # 只需把封装所在目录加入路径；封装函数会自动加入 WaveClus 子目录。
            self._engine.addpath(os.path.dirname(self.matlab_file), nargout=0)
        return self._engine

    @staticmethod
    def _normalize_paths(paths, name="paths"):
        """把单个路径或路径列表统一成字符串列表。"""
        import os

        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        else:
            paths = list(paths)
        if not paths:
            raise ValueError(name + " 不能为空")
        return [os.path.abspath(os.fspath(path)) for path in paths]

    @staticmethod
    def _to_matlab_cell(values):
        """把 Python 字符串列表转换为 MATLAB cell column。"""
        import matlab

        # 新版 Engine 提供 matlab.cell；旧版 Engine 没有该类，
        # 但会把 Python 字符串列表自动转换为 MATLAB cellstr。
        if hasattr(matlab, "cell"):
            cell = matlab.cell(len(values), 1)
            for index, value in enumerate(values):
                cell[index][0] = str(value)
            return cell
        return [str(value) for value in values]

    @classmethod
    def _to_matlab_value(cls, value):
        """递归转换 MATLAB Engine 可接受的值。"""
        import numbers
        import matlab

        if isinstance(value, dict):
            return cls._to_matlab_struct(value)
        if isinstance(value, bool):
            if hasattr(matlab, "logical"):
                return matlab.logical([value])
            return bool(value)
        if isinstance(value, numbers.Number):
            if hasattr(matlab, "double"):
                return matlab.double([float(value)])
            return float(value)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            if all(isinstance(item, str) for item in value):
                return cls._to_matlab_cell(value)
            if hasattr(matlab, "double"):
                return matlab.double([[float(item) for item in value]])
            return [[float(item) for item in value]]
        if value is None:
            return matlab.double([]) if hasattr(matlab, "double") else []
        return value

    @classmethod
    def _to_matlab_struct(cls, values):
        """创建 MATLAB struct；兼容不同版本 Engine 的赋值接口。"""
        import matlab

        if not hasattr(matlab, "struct"):
            # 极旧版本 Engine 可直接把 Python dict 转换为 MATLAB struct。
            return {str(key): cls._to_matlab_value(value)
                    for key, value in values.items()}

        struct = matlab.struct()
        for key, value in values.items():
            converted = cls._to_matlab_value(value)
            try:
                # MATLAB Engine 的 struct 通常支持字典式下标赋值。
                struct[str(key)] = converted
            except (TypeError, AttributeError):
                # 某些旧版本支持属性式赋值。
                setattr(struct, str(key), converted)
        return struct

    @classmethod
    def _to_matlab_config(cls, settings):
        """把 {'par': {...}, 'parallel': True} 转成 MATLAB config 结构体。"""
        if settings is None:
            settings = {}
        if not isinstance(settings, dict):
            raise TypeError("settings 必须是字典，例如 {'par': {'sr': 30000}}")

        return cls._to_matlab_struct(settings)

    def _call_wrapper(self, action, input_paths, output_path, settings,
                      probe_paths=None):
        """调用 MATLAB 封装函数；同一批路径只产生一次 Engine 调用。"""
        engine = self._get_engine()
        input_cell = self._to_matlab_cell(input_paths)
        probe_cell = self._to_matlab_cell(probe_paths) if probe_paths else []
        config = self._to_matlab_config(settings)
        return engine.waveclus_wrapper(
            action,
            self.root_dir,
            input_cell,
            output_path,
            config,
            probe_cell,
            nargout=1,
        )

    def extract_spikes(self, recording_paths=None, output_dir=None,
                       settings=None, probe_path=None):
        """批量调用 WaveClus 的 ``Get_spikes`` 检测连续 recording。

        该方法只启动一次 MATLAB Engine，然后把整个 ``recording_paths`` 列表
        传给 MATLAB 的一个 ``Get_spikes`` 调用。因此输入文件很多时，不会为每个
        文件重复启动 MATLAB。WaveClus 会为每个输入文件生成一个 ``*_spikes.mat``。

        ``probe_path`` 是可选的 polytrode 配置文件。普通 ``Get_spikes`` 没有
        probe 参数；提供它以后，MATLAB 封装会改用 ``Get_spikes_pol``。一个
        probe 文件描述一组需要联合检测的通道，文件每一行是一个 recording 路径，
        例如 4 通道 tetrode 可以写 4 行，8 通道组则写 8 行。组内通道数量不限，
        不要求必须是 4。若有多个通道组，``probe_path`` 传入 probe 文件列表，
        每个 probe 文件对应一组通道。

        当提供 ``probe_path`` 时，``recording_paths`` 可以省略（设为 ``None``）。
        此时程序会自动读取每个 probe 文件中的 recording 路径，并将这些路径传给
        MATLAB 封装进行文件检查。若同时提供 ``recording_paths``，它可以作为显式
        的 recording 文件列表；probe 中的路径仍会一并加入检查列表。多个 probe
        文件之间可以引用不同的 recording 文件，也可以引用同一个文件。

        Parameters
        ----------
        recording_paths : str or sequence of str, optional
            连续 recording 路径。可以是一个字符串，也可以是字符串列表。
            不使用 probe 时必须提供；列表中的每个文件会被当作一个独立 recording
            并分别调用普通 ``Get_spikes``。使用 probe 时可以不提供，程序会从
            probe 文件自动解析；如果提供，建议列出 probe 中的所有通道文件。
        output_dir : str
            ``*_spikes.mat`` 的输出目录。目录不存在时自动创建。
        settings : dict, optional
            MATLAB 参数字典，结构为：

            ``{'parallel': bool, 'par': {'参数名': 参数值, ...}}``。

            ``parallel`` 控制 WaveClus 是否并行处理多个输入文件；``par`` 中的
            字段会覆盖 ``set_parameters.m`` 的默认值。例如：

            ``{'parallel': True, 'par': {'sr': 30000, 'w_pre': 20,
            'w_post': 40, 'detection': 'neg'}}``。

            ``settings`` 不传或传 ``None`` 时，使用 WaveClus 默认值。
        probe_path : str or sequence of str, optional
            ``polytrodeN.txt`` 文件路径或路径列表。每个文件列出一组通道的
            recording 文件名。该参数只对检测阶段有效；它会触发
            ``Get_spikes_pol``，不用于后续 ``Do_clustering``。

        Returns
        -------
        list of str
            预计生成的 MAT 文件路径。

            不使用 probe 时，返回值与 ``recording_paths`` 一一对应，例如：
            ``['output/ch1_spikes.mat', 'output/ch2_spikes.mat']``。

            使用一个或多个 probe 时，每个 probe 返回一个
            ``polytrode{i}_spikes.mat``，其中 i 是 probe 文件在列表中的序号。
            这些文件的结构是：

            * ``spikes``：二维数组 ``(N, T*C)``，N 是检测到的 spike 数，T 是
              每个通道的波形采样点数，C 是该组通道数；每行按
              ``[ch1 的 T 点, ch2 的 T 点, ...]`` 排列；
            * ``index``：一维/列向量，长度 N，spike 时间，单位毫秒；
            * ``par``：WaveClus 参数结构体，其中 ``par.channels = C``；
            * 可能还有 ``thr``、``psegment``、``sr_psegment`` 等辅助变量。
        """
        import os

        probes = [] if probe_path is None else self._normalize_paths(probe_path, "probe_path")
        if probes:
            probe_recordings = self._recordings_in_probes(probes)
            if recording_paths is None:
                recordings = probe_recordings
            else:
                recordings = self._normalize_paths(recording_paths, "recording_paths")
                recordings = list(dict.fromkeys(recordings + probe_recordings))
        else:
            if recording_paths is None:
                raise ValueError(
                    "不使用 probe_path 时必须提供 recording_paths；"
                    "使用 probe 时可以省略 recording_paths。"
                )
            recordings = self._normalize_paths(recording_paths, "recording_paths")
        if output_dir is None:
            raise ValueError("output_dir 不能为空")
        output_dir = os.path.abspath(os.fspath(output_dir))
        os.makedirs(output_dir, exist_ok=True)

        for path in recordings:
            if not os.path.isfile(path):
                raise FileNotFoundError("找不到 recording 文件: " + path)
        for path in probes:
            if not os.path.isfile(path):
                raise FileNotFoundError("找不到 probe 文件: " + path)

        self._call_wrapper("get_spikes", recordings, output_dir, settings or {}, probes)

        if probes:
            return [os.path.join(output_dir, f"polytrode{i}_spikes.mat")
                    for i in range(1, len(probes) + 1)]
        return [os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0]
                             + "_spikes.mat") for path in recordings]

    @staticmethod
    def _recordings_in_probes(probe_paths):
        """读取 probe 文件中的 recording 路径并转换为绝对路径。"""
        import os

        recordings = []
        for probe_path in probe_paths:
            probe_dir = os.path.dirname(probe_path)
            with open(probe_path, "r", encoding="utf-8-sig") as file:
                for raw_line in file:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or line.startswith("%"):
                        continue
                    recording = line.strip('"\'')
                    if not os.path.isabs(recording):
                        recording = os.path.join(probe_dir, recording)
                    recordings.append(os.path.abspath(recording))
        if not recordings:
            raise ValueError("probe 文件中没有找到有效的 recording 路径")
        return list(dict.fromkeys(recordings))

    def cluster_waveforms(self, waveform_mat_paths, output_hdf5,
                          settings=None):
        """批量调用 WaveClus 的 ``Do_clustering`` 对 snippets 进行聚类。

        ``Do_clustering`` 不会读取 probe 文件；多通道信息必须已经编码在每个
        MAT 文件的 ``spikes`` 和 ``par.channels`` 中。probe 只用于连续 recording
        的检测和通道分组，应传给 :meth:`extract_spikes`。

        多通道 snippets 可以直接以 ``(N, T, C)`` 的 NumPy 数组通过
        :meth:`save_waveforms_mat` 保存。保存函数会转换成 WaveClus 的二维格式
        ``(N, T*C)``，而不是把 C 个通道平均成一个通道。C 可以是 2、4、8、16
        或其他任意正整数。

        Parameters
        ----------
        waveform_mat_paths : str or sequence of str
            snippets MAT 路径或路径列表。每个 MAT 必须至少包含：

            * ``spikes``：二维 ``(N, T)``（单通道）或 ``(N, T*C)``（多通道）；
            * ``index``：长度 N 的 spike 时间，单位毫秒；
            * ``par``：WaveClus 参数结构体；多通道时建议包含
              ``par.channels = C``。

            多个 MAT 会在一次 MATLAB Engine 会话中批量处理。
        output_hdf5 : str
            输出 HDF5 路径。父目录不存在时自动创建。多个输入文件会分别保存到
            ``/recording_1``、``/recording_2`` 等组。
        settings : dict, optional
            结构为 ``{'parallel': bool, 'par': {'参数名': 参数值, ...}}``。
            例如 ``{'parallel': True, 'par': {'min_clus': 20,
            'features': 'wav', 'max_spk': 40000}}``。不传时使用
            ``set_parameters.m`` 默认值。封装固定使用 ``make_times=True``、
            ``make_plots=False``、``save_spikes=True``。

        Returns
        -------
        str
            实际生成的 HDF5 文件绝对路径。HDF5 中每个 recording 组通常包含：

            * ``cluster_class``：二维 ``(N, 2)``，第 1 列为 cluster ID，第 2 列
              为 spike 时间（毫秒）；
            * ``index``：原始 snippets 时间戳（如果 MAT 结果中存在）；
            * ``spikes``：sorting 后保存的波形（如果 ``save_spikes=True``）。
        """
        import os

        waveform_paths = self._normalize_paths(waveform_mat_paths, "waveform_mat_paths")
        output_hdf5 = os.path.abspath(os.fspath(output_hdf5))
        os.makedirs(os.path.dirname(output_hdf5), exist_ok=True)
        for path in waveform_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError("找不到 snippets MAT 文件: " + path)

        self._call_wrapper("do_clustering", waveform_paths, output_hdf5,
                           settings or {})
        return output_hdf5

    @staticmethod
    def save_recording_mat(recording, mat_path, sr=None):
        """把连续 recording 保存成 ``Get_spikes`` 可读取的 MAT 文件。

        WaveClus 的 ``mat_wc_reader`` 会读取 MAT 中名为 ``data`` 的变量，并把它
        当作一条连续的一维信号。因此本方法只接受单通道数组。若有 C 个通道组成
        一组（C 可以是任意数量，不限于 tetrode 的 4），不能把 ``(C, N)`` 或
        ``(N, C)`` 直接保存为一个 ``data`` 矩阵再调用普通 ``Get_spikes``。
        正确做法是：

        1. 每个通道分别调用本方法保存成一个 MAT；
        2. 创建一个 ``polytrodeN.txt``，每行写一个通道 MAT 路径；
        3. 在 :meth:`extract_spikes` 中把该 probe 文件传给 ``probe_path``。

        这样 WaveClus 会在检测时联合这一组 C 个通道，并生成
        ``spikes`` 形状为 ``(N_spikes, T*C)`` 的 MAT。若 16 个通道彼此独立，
        就不要写进同一个 probe，而应分别检测和 sorting。

        Parameters
        ----------
        recording : numpy.ndarray
            单通道连续数据，形状必须是 ``(N,)``、``(N, 1)`` 或 ``(1, N)``。
            N 是连续 recording 的采样点数，不是 spike 数。数据必须是数值类型。
        mat_path : str or os.PathLike
            输出 MAT 文件路径。父目录不存在时自动创建。
        sr : float, optional
            采样率 Hz。提供后写入 MAT 的 ``sr`` 变量；不提供时必须在
            ``extract_spikes(settings={'par': {'sr': ...}})`` 中设置。

        Returns
        -------
        str
            保存后的 MAT 文件绝对路径。文件变量结构为：

            * ``data``：形状 ``(1, N)`` 的连续信号；
            * ``sr``：可选的 ``1×1`` 采样率变量。
        """
        import os
        import numpy as np
        from scipy.io import savemat

        array = np.asarray(recording)
        if array.ndim == 1:
            # mat_wc_reader 按连续一维信号读取；保存为行向量最接近 WaveClus 示例。
            array = array.reshape((1, -1))
        elif array.ndim == 2 and 1 in array.shape:
            array = array.astype(np.float64, copy=False)
        else:
            raise ValueError(
                "Get_spikes 的 MAT reader 是单通道 reader，recording 必须是 "
                "(N,), (N,1) 或 (1,N)，不能直接传入多通道矩阵。"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("recording 必须是数值 ndarray")

        mat_path = os.path.abspath(os.fspath(mat_path))
        os.makedirs(os.path.dirname(mat_path), exist_ok=True)
        payload = {"data": array.astype(np.float64, copy=False)}
        if sr is not None:
            payload["sr"] = float(sr)
        savemat(mat_path, payload, do_compression=True)
        return mat_path

    @staticmethod
    def save_waveforms_mat(waveforms, timestamps, mat_path, sr=30000,
                           timestamps_unit="samples", nbefore=20, nafter=None,
                           params=None):
        """保存 snippets 波形和时间戳为 ``Do_clustering`` 可读取的 MAT 文件。

        该方法支持单通道和任意数量 C 个通道的联合 snippets。它不会进行
        sorting，只负责把 NumPy 数据转换成 WaveClus 要求的 MAT 结构。

        Parameters
        ----------
        waveforms : numpy.ndarray
            单通道时形状为 ``(N, T)``；多通道时形状为 ``(N, T, C)``。
            N 是 spike 数，T 是每个通道的 snippet 采样点数，C 是通道数，
            可以是 2、4、8、16 或其他数量。多通道输入会转换成二维
            ``spikes``，形状为 ``(N, T*C)``，每行排列为
            ``[ch1 的 T 点, ch2 的 T 点, ..., chC 的 T 点]``，并写入
            ``par.channels = C``。不会跨通道求平均或把通道丢弃。
        timestamps : numpy.ndarray
            长度为 N 的时间戳数组。允许形状 ``(N,)``、``(N,1)`` 或 ``(1,N)``；
            单位由 ``timestamps_unit`` 指定。时间戳必须与每个 spike 一一对应。
        mat_path : str or os.PathLike
            输出 MAT 文件路径。父目录不存在时自动创建。
        sr : float
            采样率 Hz。只有当 ``timestamps_unit='samples'`` 时才用于把采样点
            换算成毫秒，但无论如何都会写入 ``par.sr``。
        timestamps_unit : {'samples', 'ms'}
            ``'samples'`` 表示 timestamps 是原始采样点；``'ms'`` 表示已经是
            毫秒。WaveClus 的 ``index`` 最终总是保存为毫秒。
        nbefore : int
            每个通道的波形峰值前采样点数。它写入 ``par.w_pre``。
        nafter : int, optional
            每个通道的波形峰值后采样点数。它写入 ``par.w_post``；不提供时按
            当前输入 T 自动推断。应保证每通道的 T 与 ``w_pre/w_post`` 约定一致。
        params : dict, optional
            写入 MAT 的 ``par`` 结构体的额外字段或覆盖字段，例如
            ``{'detection': 'neg', 'channels': 8}``。通常不需要手动设置
            ``channels``，函数会根据输入的 C 自动写入；只有确实需要覆盖时才设置。

        Returns
        -------
        str
            保存后的 MAT 文件绝对路径。MAT 文件变量结构为：

            * ``spikes``：单通道为 ``(N,T)``；C 通道为 ``(N,T*C)``；
            * ``index``：``(N,1)`` 的毫秒时间戳；
            * ``par``：MATLAB struct，至少包含 ``sr``、``w_pre``、``w_post``、
              ``channels``，以及 ``params`` 指定的字段。
        """
        import os
        import numpy as np
        from scipy.io import savemat

        waveform_array = np.asarray(waveforms)
        if waveform_array.ndim == 2:
            # 单通道：已经符合 WaveClus 的二维 spikes 格式。
            n_spikes, samples_per_channel = waveform_array.shape
            channels = 1
            spikes = waveform_array
        elif waveform_array.ndim == 3:
            # 多通道：WaveClus MAT 格式没有第三维，按通道块横向拼接。
            n_spikes, samples_per_channel, channels = waveform_array.shape
            spikes = waveform_array.transpose(0, 2, 1).reshape(
                n_spikes, channels * samples_per_channel
            )
        else:
            raise ValueError("waveforms 必须是 (N,T) 或 (N,T,C)")
        if not np.issubdtype(waveform_array.dtype, np.number):
            raise TypeError("waveforms 必须是数值 ndarray")

        times = np.asarray(timestamps).reshape(-1)
        if times.size != n_spikes:
            raise ValueError("timestamps 数量必须等于 waveforms 的第一维 N")
        if nbefore < 0 or nbefore >= samples_per_channel:
            raise ValueError("nbefore 必须在 0 到 T-1 之间")
        if nafter is None:
            nafter = samples_per_channel - nbefore - 1
        if nbefore + nafter + 1 != samples_per_channel:
            raise ValueError("nbefore + nafter + 1 必须等于波形长度 T")

        unit = str(timestamps_unit).lower()
        if unit in ("samples", "sample"):
            index_ms = times.astype(np.float64) / float(sr) * 1000.0
        elif unit in ("ms", "millisecond", "milliseconds"):
            index_ms = times.astype(np.float64)
        else:
            raise ValueError("timestamps_unit 只能是 'samples' 或 'ms'")

        par = {
            "sr": float(sr),
            "w_pre": float(nbefore),
            "w_post": float(nafter),
            "channels": int(channels),
        }
        if params:
            par.update(dict(params))

        mat_path = os.path.abspath(os.fspath(mat_path))
        os.makedirs(os.path.dirname(mat_path), exist_ok=True)
        savemat(mat_path, {
            "spikes": spikes.astype(np.float64, copy=False),
            "index": index_ms.astype(np.float64, copy=False).reshape((-1, 1)),
            "par": par,
        }, do_compression=True)
        return mat_path

    @staticmethod
    def describe_settings():
        """打印并返回 Get_spikes、Do_clustering 的设置字典。"""
        from pprint import pprint

        settings = {
            "Get_spikes": {
                "function_options": {
                    "parallel": {
                        "default": False,
                        "description": "是否并行处理多个输入文件。"
                    },
                    "par": {
                        "default": "set_parameters() 中的检测参数",
                        "description": "用于覆盖默认参数的结构体。"
                    },
                },
                "par": {
                    "sr": {"default": 10000, "description": "采样率 Hz；文件无采样率时使用。"},
                    "segments_length": {"default": 5, "description": "连续数据分段长度，分钟。"},
                    "tmin": {"default": 0, "description": "开始处理时间，秒。"},
                    "tmax": {"default": "all", "description": "结束处理时间，秒；all 表示全部。"},
                    "w_pre": {"default": 15, "description": "波形峰值前保留的采样点数。"},
                    "w_post": {"default": 25, "description": "波形峰值后保留的采样点数。"},
                    "alignment_window": {"default": 10, "description": "峰值对齐搜索窗口，采样点。"},
                    "stdmin": {"default": 3, "description": "检测阈值，噪声标准差倍数。"},
                    "stdmax": {"default": 50, "description": "过大幅度伪迹剔除阈值，噪声标准差倍数。"},
                    "detect_fmin": {"default": 300, "description": "检测高通频率 Hz。"},
                    "detect_fmax": {"default": 3000, "description": "检测低通频率 Hz。"},
                    "detect_order": {"default": 4, "description": "检测滤波器阶数；0 表示关闭。"},
                    "sort_fmin": {"default": 300, "description": "保存/对齐波形的高通频率 Hz。"},
                    "sort_fmax": {"default": 3000, "description": "保存/对齐波形的低通频率 Hz。"},
                    "sort_order": {"default": 2, "description": "保存/对齐波形滤波器阶数；0 表示关闭。"},
                    "ref_ms": {"default": 1.5, "description": "检测死区，毫秒。"},
                    "detection": {"default": "neg", "description": "检测极性：pos、neg 或 both。"},
                    "int_factor": {"default": 5, "description": "波形插值倍数。"},
                    "interpolation": {"default": "y", "description": "是否进行波形插值：y/n。"},
                },
                "probe_note": "Get_spikes 不接受 probe_path；使用 probe 时封装调用 Get_spikes_pol。",
            },
            "Do_clustering": {
                "function_options": {
                    "parallel": {"default": False, "description": "是否并行处理多个 snippets 文件。"},
                    "make_times": {"default": True, "description": "是否计算并生成 times_*.mat。封装固定为 True。"},
                    "make_plots": {"default": True, "description": "是否生成图像；封装为无 GUI 的 False。"},
                    "resolution": {"default": "-r150", "description": "批处理图像分辨率。封装关闭绘图时不使用。"},
                    "save_spikes": {"default": True, "description": "是否把波形保存到 times_*.mat。封装固定为 True。"},
                    "par": {"default": "set_parameters() 中的聚类参数", "description": "覆盖 SPC 和 template matching 参数。"},
                },
                "par": {
                    "mintemp": {"default": 0.1, "description": "SPC 最小温度。"},
                    "maxtemp": {"default": 0.251, "description": "SPC 最大温度。"},
                    "tempstep": {"default": 0.01, "description": "SPC 温度步长。"},
                    "SWCycles": {"default": 100, "description": "每个温度的 SPC 迭代次数。"},
                    "KNearNeighb": {"default": 11, "description": "SPC 近邻数。"},
                    "min_clus": {"default": 20, "description": "自动选簇时的最小簇大小。"},
                    "randomseed": {"default": 0, "description": "随机种子；0 表示使用时钟。"},
                    "c_ov": {"default": 0.7, "description": "簇重叠判定系数。"},
                    "elbow_min": {"default": 0.4, "description": "温度曲线 elbow 判定阈值。"},
                    "features": {"default": "wav", "description": "特征类型：wav 或 pca。"},
                    "scales": {"default": 4, "description": "Wavelet 分解尺度。"},
                    "min_inputs": {"default": 10, "description": "至少使用的特征数。"},
                    "max_inputs": {"default": 0.75, "description": "最多使用的特征数；小于 1 时表示比例。"},
                    "template_sdnum": {"default": 3, "description": "template matching 的标准差半径。"},
                    "template_k": {"default": 10, "description": "template matching 近邻数。"},
                    "template_k_min": {"default": 10, "description": "投票所需最少近邻数。"},
                    "template_type": {"default": "center", "description": "模板分类方法：nn/center/ml/mahal。"},
                    "force_feature": {"default": "spk", "description": "强制归类使用的特征：spk 或 wav。"},
                    "force_auto": {"default": True, "description": "是否自动执行强制归类。"},
                    "match": {"default": "y", "description": "是否启用 template matching：y/n。"},
                    "max_spk": {"default": 40000, "description": "SPC 训练前最多使用的 spike 数。"},
                    "permut": {"default": "y", "description": "超过 max_spk 时是否随机抽样：y/n。"},
                    "nbins": {"default": 100, "description": "ISI 直方图 bin 数。"},
                    "bin_step": {"default": 1, "description": "ISI 直方图步长百分比。"},
                },
                "probe_note": "Do_clustering 不读取 probe；输入 MAT 中的 par.channels 才是通道信息。",
            },
        }

        pprint(settings, sort_dicts=False, width=120)
        return settings

    def close(self):
        """关闭 MATLAB Engine。"""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    # 只打印配置，不启动 MATLAB，便于先检查当前封装支持的参数。
    WaveClusBatchSorter.describe_settings()
