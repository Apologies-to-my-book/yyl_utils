"""
尖峰排序（spike sorting）工具模块。

文件用途：
提供原始电生理数据读取、SpikeInterface 预处理、尖峰排序、结果分析、可视化、
Phy 导出和 CellExplorer 联动等功能。

整体结构：
1. launch_phy()：启动 Phy 图形界面。
2. SpikeSortingPipeline：封装完整尖峰排序工作流及各个独立处理步骤。
3. print_sorter_params() 等模块级辅助函数：提供排序器参数查询等功能。

主要执行流程：
创建 SpikeSortingPipeline 实例后调用 run_pipeline()；该方法依次保存 Recording、
应用预处理 Pipeline、运行 sorter、创建 SortingAnalyzer、生成图表并导出 Phy 数据。
get_all_preprocess_pipeline_dict() 用于查询当前 SpikeInterface 支持的全部预处理方法；
实际运行时可通过 config["preprocess_pipeline_dict"] 自定义预处理流程。
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from probeinterface import Probe
    from pathlib import Path

def launch_phy(params_path, conda_env="phy2"):
    """
    快速启动phy template-gui,注意应该输入的文件名是params.py文件

    参数:
    params_path: params.py文件路径
    conda_env: conda环境名，默认"phy2"
    """
    # # 使用PowerShell，它处理中文路径更好
    import subprocess
    cmd = f'''
    & conda activate "{conda_env}"
    & phy template-gui "{params_path}"
    '''
    subprocess.run(['powershell', '-Command', cmd])


class _PipelineOutputPaths:
    """集中管理管道各阶段的输出路径。"""
    def __init__(self, base_folder: Path):
        self.base_folder: Path = base_folder
        self.raw_recording_folder = base_folder / 'raw_recording_folder'
        self.preprocessed_recording_folder = base_folder / 'preprocessed_recording_folder'
        self.sorting_verbose_folder = base_folder / 'sorting_verbose_folder'
        self.sorting_object_folder = base_folder / 'sorting_object_folder'
        self.sorting_analyzer_folder = base_folder / 'sorting_analyzer_folder'
        self.phy_folder = base_folder / 'phy_folder'
        self.figures_folder = base_folder / 'figures_folder'
        self.qm_excel_path = base_folder / 'qm_excel.xlsx'
        self.template_metrics_path = base_folder / 'template_metrics.xlsx'
        self.cell_metrics_path = base_folder / 'cell_metrics.joblib'
        self.cell_type_metrics_path = base_folder / "cell_type_metrics.xlsx"

class _ParameterCatalogMixin:
    """按职责组织的 ParameterCatalog 操作。重型依赖保持方法内懒加载。"""
    @staticmethod
    def get_all_sorting_params_dict(sorter_name=None, print_all_sorters=True):
        """
        查询 sorter 的官方默认参数，并可同时打印全部可用 sorter。

        Parameters
        ----------
        sorter_name : str | None, default: None
            需要查询参数的 sorter 名称，例如 ``mountainsort5``。None 时只打印
            sorter 列表并返回空字典。
        print_all_sorters : bool, default: True
            是否打印全部可用和已安装的 sorter。

        Returns
        -------
        dict
            指定 sorter 的官方默认参数；未指定或查询失败时返回空字典。
        """

        def print_sorter_params(sorter_name=None):
            """
            获取某个sorting方法的详细参数和描述
            :param sorter_name: sorting方法名称，若为None则打印所有可用的sorter
            """
            from spikeinterface.sorters import get_sorter_params_description, get_default_sorter_params, available_sorters, \
                installed_sorters
            import os

            # 设置CMD为UTF-8模式
            os.system('chcp 65001')

            if sorter_name is None:
                # 打印所有可用的sorter（包括未安装的）
                available = available_sorters()
                print("\n" + "=" * 80)
                print(f"【所有可用的 Sorter】(共 {len(available)} 个):")
                print("=" * 80)
                for sorter in available:
                    print(f"  - {sorter}")

                # 打印已安装的sorter
                installed = installed_sorters()
                print("\n" + "=" * 80)
                print(f"【当前环境已安装的 Sorter】(共 {len(installed)} 个):")
                print("=" * 80)
                for sorter in installed:
                    print(f"  - {sorter}")

                print("\n" + "=" * 80)
                print("【各 Sorter 默认参数及参数说明】")
                print("=" * 80)

                for sorter_name in available:
                    print(f"\n{'=' * 60}")
                    print(f"Sorter: {sorter_name}")
                    print('-' * 60)

                    params = get_default_sorter_params(sorter_name)
                    print(f"\n【默认参数】(共 {len(params)} 个):")
                    print(params)

                    desc = get_sorter_params_description(sorter_name)
                    print(f"\n【参数说明】:")
                    for key, value in desc.items():
                        print(f"    {key}: {value}")

                    print('=' * 60)
                return

            # 打印单个sorter的参数
            print("\n" + "=" * 60)
            print(f"Sorter: {sorter_name}")
            print('-' * 60)

            params = get_default_sorter_params(sorter_name)
            print(f"\n【默认参数】(共 {len(params)} 个):")
            print(params)

            description = get_sorter_params_description(sorter_name)
            print(f"\n【参数说明】:")
            for param_name, param_desc in description.items():
                print(f"    {param_name}: {param_desc}")

            print("\n" + "=" * 60)


        
        if print_all_sorters:
            print_sorter_params()

        if sorter_name is None:
            return {}

        import spikeinterface.sorters as ss
        try:
            return ss.get_default_sorter_params(sorter_name)
        except Exception:
            print("打印sorter参数时出错，请检查环境中是否有该sorter")
            return {}

    @staticmethod
    def _get_current_sorting_params_dict(sorter_name="mountainsort5"):
        """
        返回当前管道实际使用的 sorter 参数。

        Parameters
        ----------
        sorter_name : str, default: "mountainsort5"
            sorter 名称。Mountainsort5 使用本项目固定配置；其他 sorter 返回
            当前 SpikeInterface 提供的官方默认参数。

        Returns
        -------
        dict
            可以直接传给 sorter 的参数字典。
        """
        if sorter_name != "mountainsort5":
            import spikeinterface.sorters as ss

            return ss.get_default_sorter_params(sorter_name)

        # Recording 已在 preprocess_recording() 中完成带通滤波，因此关闭 sorter
        # 内部滤波；whitening 保留在 sorter 内，并在按 shank 拆分后分别执行。
        return {
            "scheme": "2",
            "detect_threshold": 5.5,
            "detect_sign": -1,
            "detect_time_radius_msec": 0.5,
            "snippet_T1": 20,
            "snippet_T2": 20,
            "npca_per_channel": 3,
            "npca_per_subdivision": 10,
            "snippet_mask_radius": 250,
            "scheme1_detect_channel_radius": 150,
            "scheme2_phase1_detect_channel_radius": 200,
            "scheme2_detect_channel_radius": 50,
            "scheme2_max_num_snippets_per_training_batch": 200,
            "scheme2_training_duration_sec": 300,
            "scheme2_training_recording_sampling_mode": "uniform",
            "scheme3_block_duration_sec": 1800,
            "freq_min": 300,
            "freq_max": 6000,
            "filter": False,
            "whiten": True,
            "delete_temporary_recording": True,
            "pool_engine": "process",
            "n_jobs": 1,
            "chunk_duration": "1s",
            "progress_bar": True,
            "mp_context": None,
            "max_threads_per_worker": 1,
        }

    @staticmethod
    def get_all_extensions_params_dict():
        """获取默认的扩展计算字典"""
        import spikeinterface as si

        total_extension_dict = {}
        for extension_name in si.get_available_analyzer_extensions():
            total_extension_dict[extension_name] = si.get_default_analyzer_extension_params(extension_name=extension_name)
        print("所有的analyzer扩展如下：")
        print(total_extension_dict)
        return total_extension_dict

    @staticmethod
    def _get_current_extensions_params():
        """
        获取本项目默认计算的 SortingAnalyzer 扩展及其参数。

        Returns
        -------
        dict
            键为扩展名称，值为当前 SpikeInterface 版本提供的默认参数。

        Notes
        -----
        参数首先从当前安装的 SpikeInterface 动态读取，避免升级后继续使用已经
        删除或改名的参数。项目只额外修改两项：每个 Unit 最多抽取 500 个随机
        spikes，并为 templates 同时计算 average、std 和 median；其中 median
        用于计算 peak-to-peak 最大通道。
        """
        import spikeinterface as si

        extension_names = (
            "random_spikes",
            "waveforms",
            "templates",
            "noise_levels",
            "amplitude_scalings",
            "correlograms",
            "isi_histograms",
            "principal_components",
            "spike_amplitudes",
            "spike_locations",
            "template_metrics",
            "template_similarity",
            "unit_locations",
            "quality_metrics",
        )
        extension_params = {
            name: si.get_default_analyzer_extension_params(name)
            for name in extension_names
        }
        extension_params["random_spikes"]["max_spikes_per_unit"] = 2000
        extension_params["templates"]["operators"] = ["average", "std", "median"]
        extension_params["correlograms"]["bin_ms"] = 0.5
        return extension_params

    @staticmethod
    def get_all_preprocess_pipeline_dict():
        """
        获取当前 SpikeInterface 支持的全部预处理方法及参数默认值。

        参数：
            无。

        返回值：
            dict: 第一层键是可以用于 ``apply_preprocessing_pipeline()`` 的方法名；
            第二层是该方法的参数及默认值。没有默认值的必填参数用中文字符串标记。
            ``**filter_kwargs`` 等键保存底层可继续传入的扩展参数。

        实现步骤：
            1. 从 SpikeInterface 的 Pipeline 注册表读取所有受支持的方法。
            2. 使用 ``inspect.signature()`` 获取每个方法的显式参数和真实默认值。
            3. 对函数签名中的 ``**kwargs``，继续读取相应底层函数的参数。

        边界情况：
            该返回值是“全部方法参数目录”，用于查询和复制配置，不能将整个字典
            一次性传入预处理函数，因为部分方法互斥，而且部分方法需要真实的必填参数。
            SpikeInterface 的注册表属于内部 API，将来升级大版本后名称可能变化。
        """
        from copy import deepcopy
        import inspect

        from spikeinterface.core import get_noise_levels
        from spikeinterface.core.recording_tools import get_random_recording_slices
        from spikeinterface.preprocessing.detect_bad_channels import detect_bad_channels
        from spikeinterface.preprocessing.pipeline import pp_names_to_functions

        required_text = "<必填：没有默认值>"
        excluded_recording_parameters = {"recording", "parent_recording"}

        def get_explicit_defaults(function):
            """读取一个函数的显式参数，并保留继续存在的 *args/**kwargs 标记。"""
            defaults = {}
            for parameter in inspect.signature(function).parameters.values():
                if parameter.name in excluded_recording_parameters:
                    continue
                if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                    defaults[f"*{parameter.name}"] = []
                    continue
                if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    defaults[f"**{parameter.name}"] = {
                        "说明": "底层函数仍允许额外关键字参数，请查看对应函数文档"
                    }
                    continue
                if parameter.default is inspect.Parameter.empty:
                    defaults[parameter.name] = required_text
                else:
                    defaults[parameter.name] = deepcopy(parameter.default)
            return defaults

        # Pipeline 中几类 **kwargs 实际会继续传给以下底层函数。
        extra_kwargs_sources = {
            "filter_kwargs": pp_names_to_functions["filter"],
            "random_chunk_kwargs": get_random_recording_slices,
            "detect_bad_channels_kwargs": detect_bad_channels,
            "noise_levels_kwargs": get_noise_levels,
        }

        all_preprocess_pipeline = {}
        for method_name, function in pp_names_to_functions.items():
            method_parameters = {}
            for parameter in inspect.signature(function).parameters.values():
                if parameter.name in excluded_recording_parameters:
                    continue

                if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                    method_parameters[f"*{parameter.name}"] = []
                    continue

                if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    source = extra_kwargs_sources.get(parameter.name)
                    if source is None:
                        method_parameters[f"**{parameter.name}"] = {
                            "说明": "该方法允许额外关键字参数，请查看对应函数文档"
                        }
                    else:
                        expanded_parameters = get_explicit_defaults(source)
                        if parameter.name == "noise_levels_kwargs":
                            expanded_parameters["random_slices_kwargs"] = get_explicit_defaults(
                                get_random_recording_slices
                            )
                        method_parameters[f"**{parameter.name}"] = expanded_parameters
                    continue

                if parameter.default is inspect.Parameter.empty:
                    method_parameters[parameter.name] = required_text
                else:
                    method_parameters[parameter.name] = deepcopy(parameter.default)

            all_preprocess_pipeline[method_name] = method_parameters

        return all_preprocess_pipeline

    @staticmethod
    def _get_current_preprocess_pipeline_dict():
        """
        返回本项目原来实际使用的默认预处理流程。

        返回值：
            dict: 可以直接传给 ``apply_preprocessing_pipeline()`` 的管道字典。

        说明：
            原代码使用无参数的通用 ``filter()``。在 SpikeInterface 0.104.8 中，
            通用 filter 要求显式提供 margin_ms，否则会报错。因此这里使用参数等价、
            边缘长度可自动计算的 ``bandpass_filter``，滤波范围仍为 300～6000 Hz。
            注意blank_saturation的硬阈值单位和传入的recording单位相同(目前统一是uV)。
        """
        return {
            "bandpass_filter": {
                "freq_min": 300.0,
                "freq_max": 6000.0,
            },
            "center": {
                "mode": "median",
                "dtype": "float32",
            },
            "blank_saturation": {
                "abs_threshold": 500,
                "direction": "both",
            },
            "common_reference": {
                "reference": "global",
                "operator": "median",
            },
        }

    # 辅助方法
    @staticmethod
    def _filter_units(sorting, min_spikes=50, min_amplitude=50):
        """过滤峰值太少或幅度太小的单元"""
        unit_ids = sorting.unit_ids
        kept_units = []
        for unit_id in unit_ids:
            spike_count = sorting.get_unit_spike_train(unit_id).size
            if spike_count >= min_spikes:
                kept_units.append(unit_id)
        return sorting.select_units(kept_units)

    @staticmethod
    def get_probe(
        num_channels: int = 16,
        positions=None,
        shank_id=None,
        device_channel_indices=None,
    ):
        """
        创建带 shank 分类的微丝电极 Probe。

        Parameters
        ----------
        num_channels : int, default: 16
            通道数量。
        positions : array-like | None, default: None
            每个通道的二维坐标，形状为 ``(num_channels, 2)``。None 时按
            ``[(i * 300, 0) for i in range(num_channels)]`` 生成。
        shank_id : array-like | None, default: None
            每个通道所属的 shank ID，长度必须等于通道数。None 时为每个通道
            分配不同的 shank，即 ``[0, 1, ..., num_channels - 1]``。
            该映射会保存为recording的"group"属性，并在sorting的时候由
            run_sorter_by_property保存为unit的“group”属性，在创建analyzer的时候由
            method="by_property"引进到analyzer的extensions计算。
        device_channel_indices : array-like | None, default: None
            建立“Probe 物理触点 → Recording 数据列”的映射，保存的是数据列的
            数组下标而不是 channel_id。None 时使用 ``[0, 1, ..., num_channels-1]``，
            表示触点顺序与 Recording 数据列顺序完全一致；未连接的触点可设为
            ``-1``。该映射会间接影响后续所有依赖通道位置、shank 和 Probe
            信息的步骤。

        Returns
        -------
        Probe
            已设置坐标、shank ID 和设备通道索引的 Probe。
        """
        from probeinterface import Probe
        import numpy as np

        n = num_channels
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError("num_channels 必须是大于 0 的整数")

        # 默认坐标为间距 300 μm 的线性排列。
        if positions is None:
            positions = [(i * 300, 0) for i in range(n)]
        positions = np.asarray(positions, dtype="float64")
        if positions.shape != (n, 2):
            raise ValueError(
                f"positions 的形状必须是 ({n}, 2)，当前为 {positions.shape}"
            )

        # 默认每个通道独占一个 shank，为后续按 shank 硬拆分排序提供分组。
        if shank_id is None:
            shank_ids = np.arange(n)
        else:
            shank_ids = np.asarray(shank_id)
            if shank_ids.ndim != 1 or shank_ids.size != n:
                raise ValueError(f"shank_id 必须是一维序列且长度等于 {n}")

        if device_channel_indices is None:
            device_channel_indices = np.arange(n, dtype="int64")
        else:
            device_channel_indices = np.asarray(device_channel_indices)
            if device_channel_indices.ndim != 1 or device_channel_indices.size != n:
                raise ValueError(
                    "device_channel_indices 必须是一维序列且长度等于 "
                    f"{n}"
                )
            if not np.issubdtype(device_channel_indices.dtype, np.integer):
                raise TypeError("device_channel_indices 只能包含整数下标")
            device_channel_indices = device_channel_indices.astype("int64", copy=False)
            if np.any(device_channel_indices < -1):
                raise ValueError("device_channel_indices 只能是 -1 或非负整数")

            connected_indices = device_channel_indices[device_channel_indices >= 0]
            if np.unique(connected_indices).size != connected_indices.size:
                raise ValueError(
                    "device_channel_indices 中已连接触点的数据列下标不能重复"
                )

        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(
            positions=positions,
            shapes='circle',
            shape_params={'radius': 12.5},
            shank_ids=shank_ids,
        )
        probe.set_device_channel_indices(device_channel_indices)
        return probe


class _RecordingMixin:
    """按职责组织的 Recording 操作。重型依赖保持方法内懒加载。"""
    def save_traces_to_recording_file(self, traces, fs, chan_ids, outputpath,
                                      metadata_folder=None, properties=None,
                                      probe: Probe = None, n_jobs=1):
        """
        保存记录数据到二进制文件

        Args:
            traces: 神经信号数据，形状为 [时间点, 通道数] 的numpy数组，
                注意默认传入的单位是uV，注意这个单位会影响后面的阈值去噪和画波形。
            fs: 采样频率 (Hz)
            chan_ids: 通道ID列表
            outputpath: 输出文件路径
            metadata_folder: 可选的元数据文件夹路径，包含记录的元数据信息
            properties: 可选的属性字典，包含LSB等信号处理属性
            probe: 可选的探针对象，用于设置通道几何位置

        Returns:
            rec_fixed: 保存后的Recording对象
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import yyl_utils as yyl

        # 创建NumpyRecording对象，将原始数据转换为SpikeInterface格式
        rec_fixed = si.NumpyRecording(
            traces_list=[traces],
            sampling_frequency=fs,
            channel_ids=chan_ids)

        if properties:
            # 给重建的recording加上LSB等属性，_properties中包括LSB信息
            rec_fixed._properties = properties

        # 给重建的recording加上探针信息
        if probe:
            # 如果提供了探针对象，使用该探针
            # 注意：如果 Probe 中包含 device_channel_indices=-1，SpikeInterface
            # 会删除未连接触点；删除后 Probe 通道数与传入的原始数据数组不匹配，
            # 使用 in_place=True 会直接报错。
            rec_fixed.set_probe(probe, in_place=True)
        else:
            # 如果没有提供探针，创建默认的线性探针
            rec_fixed.set_probe(self.get_probe(num_channels=len(chan_ids)), in_place=True)

        if metadata_folder:
            # 给重建的recording复制metadata（如通道信息、增益等）
            rec_fixed.load_metadata_from_folder(metadata_folder)

        # 保存重建的recording到二进制文件
        yyl.check_delete_exists_path(outputpath)  # 检查并删除已存在的路径
        rec_fixed.set_property(key="group",values=[shank.shank_id for shank in probe.get_shanks()])
        rec_fixed.save(folder=outputpath, format="binary", name='plx测试', verbose=True, n_jobs=n_jobs)
        return rec_fixed

    def read_save_plx_file(self, inputpath, outputpath, probe: Probe = None, n_jobs=1):
        """
        将plx文件转换为二进制文件夹并保存
        Args:
            inputpath: 输入的plx文件路径
            outputpath: 二进制文件夹路径
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import spikeinterface.extractors as se
        import yyl_utils as yyl

        # 提取原始数据
        test_recording = se.read_plexon(inputpath, stream_id='WBC')  # 读plx文件
        # 用同样的通道属性重建recording(plx直接保存会报错)
        # 得到plx文件的原始数据,已return_in_uV,注意return_in_uV会导致cellexplorer读取失败，
        # 失败原因是cellexplorer内置有一个LSB，你缩放后的数据它还会再缩放一次导致数值过大读取失败。
        traces = test_recording.get_traces(return_in_uV=True)
        fs = test_recording.get_sampling_frequency()  # 得到plx文件的采样率
        chan_ids = test_recording.channel_ids  # 得到plx文件的通道id

        rec_fixed = si.NumpyRecording(
            traces_list=[traces],
            sampling_frequency=fs,
            channel_ids=chan_ids)

        # 给重建的recording复制metadata
        test_recording.copy_metadata(rec_fixed)
        # 给重建的recording加上LSB,_properties中包括LSB信息
        rec_fixed._properties = test_recording._properties

        # 给重建的recording加上探针信息
        if probe:
            # 如果提供了探针对象，使用该探针
            # 注意：如果 Probe 中包含 device_channel_indices=-1，SpikeInterface
            # 会删除未连接触点；删除后 Probe 通道数与传入的原始数据数组不匹配，
            # 使用 in_place=True 会直接报错。
            rec_fixed.set_probe(probe, in_place=True)
        else:
            # 如果没有提供探针，创建默认的线性探针
            rec_fixed.set_probe(self.get_probe(num_channels=len(chan_ids)), in_place=True)


        # 保存重建的recording
        yyl.check_delete_exists_path(outputpath)
        rec_fixed.set_property(key="group",values=[shank.shank_id for shank in probe.get_shanks()])
        rec_fixed.save(folder=outputpath, format="binary", name='plx测试', verbose=True, n_jobs=n_jobs)
        return rec_fixed

    def preprocess_recording(
        self,
        input_folder,
        output_folder_preprocessed,
        preprocess_pipeline_dict=None,
        n_jobs=1,
    ):
        """
        使用管道字典对原始 Recording 进行预处理。

        Args:
            input_folder: 输入的 Recording 文件夹。
            output_folder_preprocessed: 主预处理结果的保存文件夹。
            preprocess_pipeline_dict: 传给 ``apply_preprocessing_pipeline()`` 的字典。
                如果为 None，使用本项目原来的默认预处理流程；传入空字典时不执行预处理。
            n_jobs: 保存 Recording 时使用的并行任务数。

        Returns:
            预处理并保存后的 Recording。

        实现步骤：
            1. 加载原始 Recording。
            2. 应用传入的 Pipeline；未传入时应用项目默认 Pipeline。
            3. 保存主预处理结果。

        说明：
            本函数不再提供额外 whitening。Mountainsort5 的 whitening 在按 shank
            拆分后的 sorting 阶段执行，避免 whitening 跨 shank 混合通道。

        边界情况：
            preprocess_pipeline_dict 必须是字典或 None；方法名和参数是否有效由
            SpikeInterface 根据当前安装版本继续校验。
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import spikeinterface.preprocessing as spre
        import yyl_utils as yyl

        if preprocess_pipeline_dict is None:
            preprocess_pipeline_dict = self._get_current_preprocess_pipeline_dict()
        if not isinstance(preprocess_pipeline_dict, dict):
            raise TypeError("preprocess_pipeline_dict 必须是字典或 None")

        raw_recording = si.load(file_or_folder_or_dict=input_folder)
        if preprocess_pipeline_dict:
            preprocessed_recording = spre.apply_preprocessing_pipeline(
                recording=raw_recording,
                pipeline_or_dict=preprocess_pipeline_dict,
            )
        else:
            # 空字典明确表示跳过预处理，但仍然允许后续保存和排序。
            preprocessed_recording = raw_recording

        # 保存预处理后的数据
        yyl.check_delete_exists_path(output_folder_preprocessed)
        preprocessed_recording.save(
            folder=output_folder_preprocessed,
            format="binary",
            verbose=True,
            n_jobs=n_jobs
        )

        return preprocessed_recording


class _SortingMixin:
    """按职责组织的 Sorting 操作。重型依赖保持方法内懒加载。"""
    def perform_sorting(
        self,
        input_folder,
        sorter_name,
        output_folder,
        params=None,
        run_in_docker=None,
        group_n_jobs=1,
    ):
        """
        按 Probe 的 shank 分组独立执行 sorting，再合并各组结果。

        Parameters
        ----------
        input_folder : str | Path
            预处理后的 Recording 文件夹。
        sorter_name : str
            sorter 名称。
        output_folder : str | Path
            合并后 Sorting 的保存文件夹。
        params : dict | None, default: None
            sorter 参数；None 使用 _get_current_sorting_params_dict()。
        group_n_jobs : int, default: 1
            同时运行的 group sorter 数量。Kilosort4 通常使用 GPU，默认串行运行
            以避免多个任务争抢显存；确认显存充足时可设置为大于 1。

        Returns
        -------
        BaseSorting
            合并各 shank 后的 Sorting。

        Notes
        -----
        Recording.set_probe() 会根据 Probe.shank_ids 生成 ``group`` 属性。
        函数会先按该属性拆分 Recording，再为每个 group 创建独立的 sorter
        工作目录。group sorter 可并行运行；某个 group 失败或没有 Unit 时只跳过
        该 group，不会阻止其它 group 完成。最后只合并成功 group 的结果。
        """
        import spikeinterface as si
        import spikeinterface.sorters as ss
        import yyl_utils as yyl
        import numpy as np
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from joblib import Parallel, delayed
        import traceback

        recording = si.load(input_folder)

        if params is None:
            params = self._get_current_sorting_params_dict(sorter_name)
        else:
            params = params.copy()

        if not isinstance(group_n_jobs, int) or isinstance(group_n_jobs, bool):
            raise TypeError("group_n_jobs 必须是整数")
        if group_n_jobs == -1:
            import os
            group_n_jobs = os.cpu_count() or 1
        if group_n_jobs <= 0:
            raise ValueError("group_n_jobs 必须是正整数或 -1")

        if "group" not in recording.get_property_keys():
            raise ValueError(
                "Recording 缺少由 Probe.shank_ids 生成的 'group' 属性，无法按 "
                "shank 独立 sorting。请先通过 recording.set_probe() 绑定 Probe。"
            )

        output_folder = Path(output_folder)
        output_folder.parent.mkdir(parents=True, exist_ok=True)
        yyl.check_delete_exists_path(output_folder)

        group_values = recording.get_property("group")
        print(f"按 {len(set(group_values.tolist()))} 个 shank 分组独立运行 {sorter_name}")

        recording_dict = recording.split_by("group")

        # sorter 的中间结果放入临时目录；最终 Sorting 保存完成后自动清理。
        with TemporaryDirectory(
            prefix=f"{output_folder.name}_shank_sorting_",
            dir=output_folder.parent,
            ignore_cleanup_errors=True,
        ) as sorter_working_folder:
            def run_one_group(group_name, group_recording):
                """运行一个 group；异常返回给主线程，不中断其它 group。"""
                group_folder = Path(sorter_working_folder) / str(group_name)
                try:
                    print(f"开始处理 group={group_name}")
                    group_sorting = ss.run_sorter(
                        sorter_name=sorter_name,
                        recording=group_recording,
                        folder=str(group_folder),
                        verbose=False,
                        docker_image=run_in_docker,
                        **params,
                    )
                    if group_sorting is None or len(group_sorting.unit_ids) == 0:
                        return group_name, None, "没有检测到有效 Unit"
                    return group_name, group_sorting, None
                except Exception as exc:
                    return group_name, None, (
                        f"{exc}\n{traceback.format_exc()}"
                    )

            effective_n_jobs = min(group_n_jobs, max(1, len(recording_dict)))
            # joblib 使用独立进程执行 group sorter。每个任务都有独立输出目录，
            # 单个任务的异常会由 run_one_group 转换成返回值，不会中断其它任务。
            parallel_results = Parallel(
                n_jobs=effective_n_jobs,
                backend="loky",
            )(
                delayed(run_one_group)(group_name, group_recording)
                for group_name, group_recording in recording_dict.items()
            )

            group_results = {}
            for group_name, group_sorting, error_message in parallel_results:
                if group_sorting is None:
                    print(f"group={group_name} 已跳过：{error_message}")
                else:
                    print(
                        f"group={group_name} 完成，Unit 数量："
                        f"{len(group_sorting.unit_ids)}"
                    )
                group_results[group_name] = group_sorting

            sorting_list = [
                group_results[group_name]
                for group_name in recording_dict
                if group_results.get(group_name) is not None
            ]
            successful_groups = [
                group_name
                for group_name in recording_dict
                if group_results.get(group_name) is not None
            ]

            if not sorting_list:
                print("所有 group 都没有产生有效 Unit，当前 sorting 结果为空。")
                empty_sorting = si.NumpySorting.from_unit_dict(
                    {}, recording.sampling_frequency
                )
                empty_sorting.save(folder=output_folder, overwrite=True)
                return empty_sorting

            # 合并成功 group 的 sorting，并恢复每个 Unit 所属的 group 属性。
            sorting = si.aggregate_units(sorting_list)
            unit_groups = []
            for group_name, group_sorting in zip(successful_groups, sorting_list):
                unit_groups.extend([group_name] * len(group_sorting.unit_ids))
            sorting.set_property("group", np.asarray(unit_groups))
            sorting.register_recording(recording)

            # run_sorter_by_property() 会把每个 Unit 所属的 group 保存为
            # Sorting 的 group property。先读取该 property，再在每个 group 内
            # 从 1 开始编号，避免所有 shank 共用 raw_unit_1、raw_unit_2 这种编号。
            if len(sorting.unit_ids) > 0:
                unit_groups = sorting.get_property("group")
                if unit_groups is None:
                    raise ValueError(
                        "sorting 结果缺少 'group' 属性，无法按 group 重命名 Unit。"
                    )
                if len(unit_groups) != len(sorting.unit_ids):
                    raise ValueError(
                        "sorting 的 'group' 属性长度与 unit_ids 数量不一致，"
                        "无法安全重命名 Unit。"
                    )

                group_counts = {}
                new_ids = []
                for group in unit_groups:
                    group_name = str(group)
                    group_counts[group_name] = group_counts.get(group_name, 0) + 1
                    # 例如 group=0 时生成 0_unit_1、0_unit_2；
                    # group=shankA 时生成 shankA_unit_1、shankA_unit_2。
                    new_ids.append(f"group_{group_name}_unit_{group_counts[group_name]}")

                sorting = sorting.rename_units(new_ids)
            sorting = sorting.save(folder=output_folder, overwrite=True)
            return sorting

    def batch_sorting(self, input_folder_list: list[str], output_base_folder, sorter_name, params=None, run_in_docker=None, n_jobs=1):
        """
        批量排序
        注意保存的verbose_path路径不能在同一个子文件夹下，必须是不同的倒数第二级文件夹，不然它们的元文件会串起来进而报错
        Args:
            input_folder_list: 总文件夹，要求文件夹下是一组预处理后的 Recording
            output_base_folder: 输出结果文件夹，包括verbose和sorting结果
            sorter_name: 排序器名称
            params: 排序参数

        Notes
        -----
        SpikeInterface 0.104.8 的 run_sorter_jobs() 没有 grouping_property 参数。
        为保留一次性批量调度及 WSL/容器效率，本函数不拆分 shank，也不改为
        Python 循环；需要按 shank 硬隔离时请使用 perform_sorting()。
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import spikeinterface.sorters as ss
        import yyl_utils as yyl

        if params is None:
            params = self._get_current_sorting_params_dict(sorter_name)

        job_list = []
        for index, folder_path in enumerate(input_folder_list):
            recording = si.load(folder_path)
            # verbose_path = output_base_folder / f"{index}" / f'verbose{index}'
            # yyl.check_delete_exists_path([verbose_path])

            job_dict = {
                'sorter_name': sorter_name,
                'recording': recording.clone(),
                # 'folder': verbose_path,
                'delete_output_folder': True,
                'verbose': True,
                'remove_existing_folder': False,
                'raise_error': False,
                'docker_image': run_in_docker,
                'delete_container_files': False,
                # 'installation_mode': "folder",
                # 'spikeinterface_folder_source':  # spikeinterface安装包路径
                #     r"C:\Users\32707\Desktop\工作\
                #     用户实验\张刘馨黛-数据分析\测试代码\安装包\spikeinterface-main\spikeinterface-main",
            }
            job_dict.update(params)
            job_list.append(job_dict)

        # 运行批量任务
        sortings = ss.run_sorter_jobs(
            job_list=job_list,
            engine='joblib',
            engine_kwargs={'n_jobs': n_jobs},
            return_output=True
        )

        if sortings is not None:
            for i, sorting in enumerate(sortings):
                # 给各个unit添加前缀 "raw_unit_"
                if len(sorting.unit_ids) > 0:
                    new_ids = [f"raw_unit_{j}" for j in range(1, len(sorting.unit_ids) + 1)]
                    sorting = sorting.rename_units(new_ids)
                    sortings[i] = sorting  # ← 更新列表

                # 保存到 output_base_folder 下的 sorting_results 文件夹
                save_folder = output_base_folder / f"sorting_{i}"  # ← 保存位置
                yyl.check_delete_exists_path(save_folder)
                yyl.make_sure_folder_exist(save_folder)
                sorting.save(folder=save_folder, overwrite=True)
        return sortings


class _AnalyzerMixin:
    """按职责组织的 Analyzer 操作。重型依赖保持方法内懒加载。"""
    def create_analyzer(self, recording_folder, sorting_folder, output_folder, extensions_dict=None,
                        compute=True, template_metrics_path=None, qm_path=None, n_jobs=1):
        """
        创建 SortingAnalyzer、计算扩展并导出指标表格。

        Parameters
        ----------
        recording_folder : str | Path
            预处理后的 Recording 文件夹。
        sorting_folder : str | Path
            spike sorting 结果文件夹。
        output_folder : str | Path
            SortingAnalyzer 的保存文件夹。
        extensions_dict : dict | None, default: None
            需要计算的扩展及参数；None 使用 _get_current_extensions_params()。
        compute : bool, default: True
            是否立即计算扩展。False 时仅创建 Analyzer。
        template_metrics_path : str | Path | None, default: None
            template_metrics Excel 输出路径。表格会增加 ``max_channel`` 列，
            该列使用 median template 的 peak-to-peak 最大通道。
        qm_path : str | Path | None, default: None
            quality_metrics Excel 输出路径。
        n_jobs : int, default: 1
            扩展计算使用的并行任务数；-1 表示使用全部可用核心。

        Returns
        -------
        SortingAnalyzer
            已创建的 Analyzer。计算 template_metrics 时，其结果中也会保存
            ``max_channel`` 列。
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        from spikeinterface.core import get_template_extremum_channel
        import yyl_utils as yyl
        from pathlib import Path

        if extensions_dict is None:
            extensions_dict = self._get_current_extensions_params()

        yyl.check_delete_exists_path(output_folder)

        # 读取recording和sorting
        recording = si.load(recording_folder)
        sorting = si.load(sorting_folder)

        # 过滤单元
        # sorting = self._filter_units(sorting)

        # 创建分析器
        analyzer = si.create_sorting_analyzer(
            sorting,
            recording,
            format="binary_folder",
            folder=output_folder,
            sparse=True,
            method="by_property",
            by_property="group",
        )

        # 一次性计算所有扩展
        if compute:
            analyzer.compute(extensions_dict, n_jobs=n_jobs)

            if analyzer.has_extension("template_metrics"):
                if not analyzer.has_extension("templates"):
                    raise ValueError(
                        "计算 max_channel 需要 templates 扩展，请将 templates 加入 "
                        "extensions_dict"
                    )

                # 使用官方接口按中位模板的峰峰值选择每个 Unit 的最大通道。
                # operator="median" 要求 templates 中存在 median；默认配置会提前
                # 计算它，自定义配置则需要同时保留 waveforms 以便按需计算。
                try:
                    peak_channels = get_template_extremum_channel(
                        analyzer,
                        peak_sign="both",
                        mode="peak_to_peak",
                        outputs="id",
                        # operator="median",
                    )
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        "无法用 median template 计算 max_channel。请确保 "
                        "extensions_dict 同时计算 random_spikes、waveforms 和 "
                        "templates，并在 templates 的 operators 中加入 'median'。"
                    ) from exc

                template_metrics_extension = analyzer.get_extension("template_metrics")
                template_metrics = template_metrics_extension.get_data().copy()
                template_metrics["max_channel"] = [
                    peak_channels[unit_id] for unit_id in template_metrics.index
                ]

                # 将自定义列写回扩展并保存，使重新加载 Analyzer 后仍能读取。
                template_metrics_extension.data["metrics"] = template_metrics
                template_metrics_extension.save()

                if template_metrics_path is not None:
                    template_metrics_path = Path(template_metrics_path)
                    template_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    template_metrics.to_excel(template_metrics_path, index=True)

            if qm_path is not None and analyzer.has_extension("quality_metrics"):
                qm_path = Path(qm_path)
                qm_path.parent.mkdir(parents=True, exist_ok=True)
                analyzer.get_extension("quality_metrics").get_data().to_excel(
                    qm_path,
                    index=True,
                )

        return analyzer


class _UnitAnalysisMixin:
    """按职责组织的 UnitAnalysis 操作。重型依赖保持方法内懒加载。"""
    def renew_unit_type(self, analyzer_folder, cell_type_metrics_path, classify_units=True):
        """
        判断细胞的分类情况及细胞的可能类型，将结果保存至路径self.output_paths.cell_type_metrics_path
        :param classify_units: 是否判断细胞类型
        :return: None
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import pandas as pd
        import numpy as np

        def screen_units(df_qm_metrics, unit_ids=None):
            """
            根据质量指标筛选符合分析条件的神经元单元
            """
            # 读取质量指标和模板指标数据
            # 初始化筛选字典：包含所有unit_id和空的质量列表
            # 以 Analyzer 的完整 Unit 列表为准。quality_metrics 缺失某些 Unit
            # 时，也要在输出表中保留这些 Unit，并将对应指标留空。
            if unit_ids is None:
                unit_ids = df_qm_metrics.index.tolist()
            else:
                unit_ids = list(unit_ids)
            screen_dict = {"unit_ids": unit_ids,
                           "sorting_quality": []}

            # Allen Visual Coding Neuropixels 论文展示 unit 数量时使用：
            # `isi_violations < 0.5`、`amplitude_cutoff < 0.1`、`presence_ratio > 0.95`，
            # 这是一个较为认可的默认标准，参考文献doi：10.1038/s41586-020-03171-x。
            isi_column = (
                "isi_violations"
                if "isi_violations" in df_qm_metrics.columns
                else "isi_violations_ratio"
            )
            for unit_id in screen_dict["unit_ids"]:
                # 按 ISI 违例率、振幅截断率和存在率判定排序质量。
                required_columns = (
                    isi_column,
                    "amplitude_cutoff",
                    "presence_ratio",
                )
                values_available = all(
                    unit_id in df_qm_metrics.index
                    and column in df_qm_metrics.columns
                    and pd.notna(df_qm_metrics.loc[unit_id, column])
                    for column in required_columns
                )
                if not values_available:
                    # 找不到某个质量指标时不臆测好坏，Excel 中留空。
                    screen_dict["sorting_quality"].append("")
                elif (
                    (df_qm_metrics.loc[unit_id, isi_column] < 0.5)
                    and (df_qm_metrics.loc[unit_id, "amplitude_cutoff"] < 0.1)
                    and (df_qm_metrics.loc[unit_id, "presence_ratio"] > 0.95)
                ):
                    # 符合所有筛选条件的单元标记为"good"
                    screen_dict["sorting_quality"].append("good")
                else:
                    screen_dict["sorting_quality"].append("bad")
            return screen_dict

        def _fit_acg_tau_rise_from_correlograms(analyzer, unit_ids):
            """按照 CellExplorer 的三指数模型，从 correlograms 拟合 tau rise。

            CellExplorer 的 ``fit_ACG.m`` 使用正半轴 ACG（0.5--50 ms，0.5 ms
            bin），并拟合
            ``max(c*(exp(-(x-f)/a)-d*exp(-(x-f)/b)) +
            h*exp(-(x-f)/g) + e, 0)``。
            SpikeInterface 返回的是 bin edges 和 ``(unit, unit, bin)`` 计数，
            因此这里取零点右侧的 bin，第一格置零以去除自相关零点峰。
            """
            import numpy as np
            import pandas as pd

            extension_name = None
            for candidate in ("correlograms", "auto_correlograms"):
                if analyzer.has_extension(candidate):
                    extension_name = candidate
                    break
            if extension_name is None:
                return pd.Series(np.nan, index=unit_ids, dtype="float64")

            try:
                ccgs, bins = analyzer.get_extension(extension_name).get_data()
                ccgs = np.asarray(ccgs)
                bins = np.asarray(bins, dtype=float)
                if ccgs.ndim != 3 or bins.ndim != 1 or ccgs.shape[-1] != bins.size - 1:
                    raise ValueError("correlograms 的数组形状与 bins 不匹配")
            except Exception as exc:
                print(f"无法读取或拟合 correlograms，acg_tau_rise 留空：{exc}")
                return pd.Series(np.nan, index=unit_ids, dtype="float64")

            # 与 CellExplorer 的初值、下界和上界保持一致，参数单位均为 ms。
            initial = np.array([20.0, 1.0, 30.0, 2.0, 0.5, 5.0, 1.5, 2.0])
            lower = np.array([1.0, 0.1, 0.0, 0.0, -30.0, 0.0, 0.1, 0.0])
            upper = np.array([500.0, 50.0, 500.0, 15.0, 50.0, 20.0, 5.0, 100.0])

            def triple_exponential(x, a, b, c, d, e, f, g, h):
                value = c * (np.exp(-(x - f) / a) - d * np.exp(-(x - f) / b))
                value += h * np.exp(-(x - f) / g) + e
                return np.maximum(value, 0.0)

            try:
                from scipy.optimize import differential_evolution, least_squares
            except Exception as exc:
                print(f"SciPy 优化器不可用，acg_tau_rise 留空：{exc}")
                return pd.Series(np.nan, index=unit_ids, dtype="float64")

            def fit_one_acg(x, y, seed):
                """先用差分进化找全局初值，再用 least_squares 精细拟合。"""
                def residual(parameters):
                    return triple_exponential(x, *parameters) - y

                def objective(parameters):
                    residual_values = residual(parameters)
                    return float(np.sum(residual_values * residual_values))

                # 差分进化降低单一起始点陷入局部最优的概率；关闭内部 polish，
                # 由后面的 least_squares 统一完成局部精修。
                de_result = differential_evolution(
                    objective,
                    bounds=list(zip(lower, upper)),
                    seed=seed,
                    maxiter=40,
                    popsize=6,
                    tol=0.02,
                    polish=False,
                    workers=1,
                    updating="immediate",
                )
                local_result = least_squares(
                    residual,
                    x0=np.clip(de_result.x, lower, upper),
                    bounds=(lower, upper),
                    method="trf",
                    loss="linear",
                    x_scale="jac",
                    max_nfev=5000,
                )
                if not local_result.success or not np.all(np.isfinite(local_result.x)):
                    return None
                return local_result.x

            bin_width = float(np.median(np.diff(bins)))
            bin_centers = (bins[:-1] + bins[1:]) / 2.0
            positive_indices = np.flatnonzero(bin_centers >= 0.0)
            tau_rise = pd.Series(np.nan, index=unit_ids, dtype="float64")

            for unit_position, unit_id in enumerate(unit_ids):
                try:
                    if unit_position >= ccgs.shape[0]:
                        continue
                    # 只使用零点右侧，x 从一个 bin 宽度开始，和 fit_ACG.m 的
                    # x = 0.5:0.5:50 写法一致；第一格对应被清零的中心 bin。
                    y = np.asarray(ccgs[unit_position, unit_position, positive_indices], dtype=float)
                    if y.size < 8:
                        continue
                    y = y.copy()
                    y[0] = 0.0
                    x = np.arange(1, y.size + 1, dtype=float) * bin_width
                    valid = np.isfinite(x) & np.isfinite(y)
                    if valid.sum() < 8 or np.nanmax(y[valid]) <= 0:
                        continue
                    fitted = fit_one_acg(x[valid], y[valid], seed=unit_position)
                    if fitted is not None and np.isfinite(fitted[1]):
                        tau_rise.loc[unit_id] = float(fitted[1])
                except Exception:
                    # 单个 Unit 的 ACG 质量不足或拟合不收敛时，不影响其它 Unit。
                    continue
            return tau_rise

        def _get_acg_tau_rise(analyzer, unit_ids, df_qm_metrics):
            """读取已有 acg_tau_rise；没有时从 correlograms 自动拟合。"""
            # 某些外部/CellExplorer 结果会直接合并到 quality_metrics 表中。
            if "acg_tau_rise" in df_qm_metrics.columns:
                return df_qm_metrics["acg_tau_rise"].reindex(unit_ids)

            # SpikeInterface 0.104.8 的默认扩展没有名为 acg_tau_rise 的标准扩展，
            # 因此兼容任意已保存扩展，只要其 get_data() 返回该列/字段即可。
            extension_names = set(analyzer.get_loaded_extension_names())
            if analyzer.format != "memory":
                extension_names.update(analyzer.get_saved_extension_names())

            for extension_name in extension_names:
                extension = analyzer.get_extension(extension_name)
                if extension is None:
                    continue
                try:
                    data = extension.get_data()
                except Exception:
                    continue

                if isinstance(data, pd.DataFrame) and "acg_tau_rise" in data.columns:
                    return data["acg_tau_rise"].reindex(unit_ids)
                if isinstance(data, dict) and "acg_tau_rise" in data:
                    values = data["acg_tau_rise"]
                    if isinstance(values, dict):
                        return pd.Series(values, dtype="float64").reindex(unit_ids)
                    return pd.Series(values, index=unit_ids, dtype="float64")

            fitted_tau_rise = _fit_acg_tau_rise_from_correlograms(analyzer, unit_ids)
            if fitted_tau_rise.notna().any():
                print("已根据 correlograms 的 ACG 直方图拟合 acg_tau_rise。")
            else:
                print("未找到或无法拟合 acg_tau_rise，相关 putative_cell_type 结果留空。")
            return fitted_tau_rise

        def get_units_classified(
            df_qm_metrics,
            df_template_metrics,
            analyzer,
            acg_tau_rise=None,
            unit_ids=None,
        ):
            """
            根据 CellExplorer 的启发式规则进行推定细胞类型分类：
            - troughToPeak <= 0.425 ms：Narrow Interneuron；
            - troughToPeak > 0.425 ms 且 acg_tau_rise > 6 ms：Wide Interneuron；
            - troughToPeak > 0.425 ms 且 acg_tau_rise <= 6 ms：Pyramidal Cell。

            ``troughToPeak`` 在 SpikeInterface 模板指标中对应
            ``peak_to_trough_duration``（旧版本为 ``peak_to_valley``），单位为秒；
            ``acg_tau_rise`` 从 Analyzer 扩展或质量指标表中读取，单位为毫秒。
            """
            # 以 Analyzer 的 Unit 列表为准；某个扩展缺失 Unit 时，该 Unit 的
            # 分类结果留空，而不是让整个 renew_unit_type 失败。
            classify_unit_ids = list(
                sorting_analyzer.unit_ids
                if unit_ids is None
                else unit_ids
            )
            classify_dict = {"unit_ids": classify_unit_ids,
                             "putative_cell_type": []}

            # 读取 ACG 拟合得到的 tau rise；该值不是 correlograms 直方图本身，
            # 必须来自额外的 ACG 拟合结果。调用方已经计算过时直接复用，
            # 避免为了“输出指标”和“细胞分类”重复拟合。
            if acg_tau_rise is None:
                acg_tau_rise = _get_acg_tau_rise(
                    analyzer,
                    classify_unit_ids,
                    df_qm_metrics,
                )

            # 判断神经元的推定类型
            if classify_dict["unit_ids"]:
                for unit_id in classify_dict["unit_ids"]:
                    # 新版 SpikeInterface 使用 peak_to_trough_duration；兼容旧表格的
                    # peak_to_valley 列，二者含义都是波峰到波谷的时间间隔（秒）。
                    if unit_id not in df_template_metrics.index:
                        classify_dict["putative_cell_type"].append("")
                        continue
                    if "peak_to_trough_duration" in df_template_metrics.columns:
                        duration_column = "peak_to_trough_duration"
                    elif "peak_to_valley" in df_template_metrics.columns:
                        duration_column = "peak_to_valley"
                    else:
                        classify_dict["putative_cell_type"].append("")
                        continue
                    trough_to_peak_ms = (
                        df_template_metrics.loc[unit_id, duration_column] * 1000
                    )
                    tau_rise_ms = acg_tau_rise.loc[unit_id]

                    try:
                        valid_metrics = np.isfinite(float(trough_to_peak_ms)) and np.isfinite(
                            float(tau_rise_ms)
                        )
                    except (TypeError, ValueError):
                        valid_metrics = False

                    if not valid_metrics:
                        # 找不到必要指标时不强行归类，Excel 中留空。
                        classify_dict["putative_cell_type"].append("")
                    elif trough_to_peak_ms <= 0.425:
                        classify_dict["putative_cell_type"].append(
                            "putative_narrow_interneuron_units"
                        )
                    elif tau_rise_ms > 6:
                        classify_dict["putative_cell_type"].append(
                            "putative_wide_interneuron_units"
                        )
                    else:
                        classify_dict["putative_cell_type"].append(
                            "putative_pyramidal_units"
                        )
            return classify_dict

        sorting_analyzer = si.load(analyzer_folder)
        # 读取质量指标Excel文件
        if sorting_analyzer.has_extension("quality_metrics"):
            df_qm_metrics = sorting_analyzer.get_extension("quality_metrics").get_data()
        else:
            print("Analyzer 缺少 quality_metrics，sorting_quality 结果留空。")
            df_qm_metrics = pd.DataFrame(index=sorting_analyzer.unit_ids)
        unit_ids = list(sorting_analyzer.unit_ids)
        # 执行单元筛选，获取质量标记；即使质量扩展不完整，也保留所有 Unit。
        renew_dict = screen_units(df_qm_metrics, unit_ids=unit_ids)

        # 将质量判断所依据的原始指标一并写入 Excel，便于回溯每个 Unit
        # 为什么被标记为 good/bad。找不到的值统一留空（NaN）。
        isi_column = (
            "isi_violations"
            if "isi_violations" in df_qm_metrics.columns
            else "isi_violations_ratio"
            if "isi_violations_ratio" in df_qm_metrics.columns
            else "isi_violations"
        )
        renew_dict[isi_column] = [
            df_qm_metrics.loc[unit_id, isi_column]
            if unit_id in df_qm_metrics.index and isi_column in df_qm_metrics.columns
            else np.nan
            for unit_id in unit_ids
        ]
        for metric_name in ("amplitude_cutoff", "presence_ratio"):
            renew_dict[metric_name] = [
                df_qm_metrics.loc[unit_id, metric_name]
                if unit_id in df_qm_metrics.index and metric_name in df_qm_metrics.columns
                else np.nan
                for unit_id in unit_ids
            ]

        # ACG 拟合指标独立于细胞类型分类输出。这样即使调用者把
        # classify_units 设为 False，Excel 中仍会保留每个 Unit 的 acg_tau_rise。
        acg_tau_rise = _get_acg_tau_rise(
            sorting_analyzer,
            unit_ids,
            df_qm_metrics,
        )
        renew_dict["acg_tau_rise"] = [
            acg_tau_rise.get(unit_id, np.nan) for unit_id in unit_ids
        ]

        # 输出细胞类型判断使用的 trough-to-peak（毫秒）。SpikeInterface
        # 通常以秒保存该指标，因此写入 Excel 前统一换算为 ms。
        if sorting_analyzer.has_extension("template_metrics"):
            df_template_metrics = sorting_analyzer.get_extension("template_metrics").get_data()
        else:
            print("Analyzer 缺少 template_metrics，troughToPeak 和 putative_cell_type 结果留空。")
            df_template_metrics = pd.DataFrame(index=unit_ids)
        duration_column = (
            "peak_to_trough_duration"
            if "peak_to_trough_duration" in df_template_metrics.columns
            else "peak_to_valley"
            if "peak_to_valley" in df_template_metrics.columns
            else None
        )
        renew_dict["troughToPeak"] = [
            df_template_metrics.loc[unit_id, duration_column] * 1000
            if duration_column is not None
            and unit_id in df_template_metrics.index
            and pd.notna(df_template_metrics.loc[unit_id, duration_column])
            else np.nan
            for unit_id in unit_ids
        ]

        # 如果需要分类神经元类型
        if classify_units:
            # 执行细胞类型分类
            classify_dict = get_units_classified(
                df_qm_metrics,
                df_template_metrics,
                sorting_analyzer,
                acg_tau_rise=acg_tau_rise,
                unit_ids=unit_ids,
            )
            # 合并筛选字典和分类字典
            renew_dict = {**renew_dict, **classify_dict}

        # 将合并后的字典转换为DataFrame并保存为Excel文件
        df_renew = pd.DataFrame.from_dict(renew_dict)
        df_renew.to_excel(cell_type_metrics_path)

    def perform_curation(self, analyzer_folder, model_folder):
        """
        使用大模型进行curation
        输出结果是给analyser的sorting打上标签"quality_noise""ratio_noise""quality_mua""ratio_mua"，并保存回原路径
        Args:
            analyzer_folder: 分析器文件夹路径
            model_folder: 模型文件夹路径
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import spikeinterface.curation as sc
        import yyl_utils as yyl

        sorting_analyzer = si.load(analyzer_folder)

        labels_noise = sc.auto_label_units(
            sorting_analyzer=sorting_analyzer,
            # repo_id="SpikeInterface/UnitRefine_noise_neural_classifier",
            model_folder=model_folder / "UnitRefine_noise_neural_classifier",
            trust_model=True
        )

        labels_sua_mua = sc.auto_label_units(
            sorting_analyzer=sorting_analyzer,
            # repo_id="SpikeInterface/UnitRefine_sua_mua_classifier",
            model_folder=model_folder / "UnitRefine_sua_mua_classifier",
            trust_model=True
        )

        # 添加属性
        sorting_analyzer.sorting.set_property("quality_noise", labels_noise.iloc[:, 0])
        sorting_analyzer.sorting.set_property("ratio_noise", labels_noise.iloc[:, 1])
        sorting_analyzer.sorting.set_property("quality_mua", labels_sua_mua.iloc[:, 0])
        sorting_analyzer.sorting.set_property("ratio_mua", labels_sua_mua.iloc[:, 1])

        # 先保存到同级临时目录；保存成功后再替换原 Analyzer，避免保存失败时
        # 原分析结果已经被删除。若替换过程失败，则尝试恢复原目录。
        from pathlib import Path
        import shutil
        import tempfile
        import uuid

        analyzer_path = Path(analyzer_folder)
        analyzer_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = Path(
            tempfile.mkdtemp(
                prefix=f"{analyzer_path.name}_updated_",
                dir=analyzer_path.parent,
            )
        )
        # save_as 要求目标目录不存在。
        shutil.rmtree(temporary_path)
        backup_path = analyzer_path.parent / (
            f"{analyzer_path.name}_backup_{uuid.uuid4().hex}"
        )
        replacement_succeeded = False

        try:
            sorting_analyzer.save_as(
                format="binary_folder",
                folder=temporary_path,
            )
            if analyzer_path.exists():
                shutil.move(str(analyzer_path), str(backup_path))
            shutil.move(str(temporary_path), str(analyzer_path))
            replacement_succeeded = True
        except Exception:
            # 新目录如果已经移动到目标位置，先移除它，再恢复旧目录。
            if analyzer_path.exists() and backup_path.exists():
                shutil.rmtree(analyzer_path)
            if backup_path.exists() and not analyzer_path.exists():
                shutil.move(str(backup_path), str(analyzer_path))
            raise
        finally:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)
            if replacement_succeeded and backup_path.exists():
                shutil.rmtree(backup_path)

        print("Noise labels:")
        print(labels_noise)
        print("SUA/MUA labels:")
        print(labels_sua_mua)

        return labels_noise, labels_sua_mua


class _VisualizationMixin:
    """按职责组织的 Visualization 操作。重型依赖保持方法内懒加载。"""
    def visualize_results(
        self,
        analyzer_folder,
        fig_folder=None,
        max_waveforms_html=2000,
        max_waveforms_png=2000,
        plot_group_waveforms=True,
        max_waveforms_group=2000,
    ):
        """
        为每个 Unit 生成 waveform、自相关及二者合并图，并可按 group 汇总波形。

        波形图只展示峰峰值最大的通道。横轴是相对 spike 对齐点的时间，单位为
        毫秒；纵轴是绝对电压，单位为微伏。PNG 中 waveform 子图的纵轴固定为
        -200 到 200 μV；HTML 保留自动缩放。每个 Unit 只保留两个 HTML 和一个 PNG；
        另外可按 group 生成汇总 waveform HTML。

        Parameters
        ----------
        analyzer_folder : str | Path
            已计算 ``waveforms`` 和 ``correlograms`` 扩展的 Analyzer 文件夹。
        fig_folder : str | Path | None, default: None
            图表输出目录；None 表示只打印 Unit 数量，不生成文件。
        max_waveforms_html : int, default: 2000
            每个 Unit 在普通 waveform HTML 中最多展示的单次波形数。Plotly 会把
            这些波形合并成一个 WebGL 数据对象；如果显卡性能有限，仍可将该值调小。
        max_waveforms_png : int, default: 2000
            合并 PNG 最多展示的单次波形数。
        plot_group_waveforms : bool, default: True
            是否将同一 group 内各 Unit 的峰值通道单次波形绘制到同一个 Plotly HTML 图中。
        max_waveforms_group : int, default: 2000
            每个 Unit 在 group waveform HTML 和 group PCA HTML 中最多使用的波形数。

        Returns
        -------
        None

        Notes
        -----
        中位波形始终使用 Analyzer 中该 Unit 的全部 waveforms 计算，不受展示数量限制。
        HTML 使用 Plotly WebGL，并把多条 waveform 合并为一个 trace，避免 mpld3
        为 2000 条曲线创建大量 SVG 对象。PNG 是静态图片，最多保留 2000 条。
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        import spikeinterface as si
        from pathlib import Path

        if not isinstance(max_waveforms_html, int) or max_waveforms_html <= 0:
            raise ValueError("max_waveforms_html 必须是大于 0 的整数")
        if not isinstance(max_waveforms_png, int) or max_waveforms_png <= 0:
            raise ValueError("max_waveforms_png 必须是大于 0 的整数")
        if not isinstance(max_waveforms_group, int) or max_waveforms_group <= 0:
            raise ValueError("max_waveforms_group 必须是大于 0 的整数")

        analyzer = si.load(analyzer_folder)
        unit_counts = analyzer.sorting.count_num_spikes_per_unit()
        print("unit_id:spike数")
        print({f"{unit_id}": int(count) for unit_id, count in unit_counts.items()})

        if fig_folder is None or len(analyzer.unit_ids) == 0:
            return

        fig_folder = Path(fig_folder)
        html_folder = fig_folder / "html"
        fig_folder.mkdir(parents=True, exist_ok=True)
        html_folder.mkdir(parents=True, exist_ok=True)

        # 每类图单独检查所需扩展。缺少某个扩展时只跳过对应图，不影响其它图。
        waveform_extension = (
            analyzer.get_extension("waveforms") if analyzer.has_extension("waveforms") else None
        )
        correlogram_extension = (
            analyzer.get_extension("correlograms")
            if analyzer.has_extension("correlograms")
            else None
        )
        all_correlograms = None
        correlogram_bins_ms = None
        if correlogram_extension is not None:
            try:
                all_correlograms, correlogram_bins_ms = correlogram_extension.get_data()
            except Exception as exc:
                print(f"读取 correlograms 扩展失败，跳过自相关图：{exc}")
                correlogram_extension = None

        if waveform_extension is None and correlogram_extension is None:
            print("Analyzer 没有 waveforms 或 correlograms 扩展，跳过全部可视化。")
            return

        principal_components_extension = (
            analyzer.get_extension("principal_components")
            if analyzer.has_extension("principal_components")
            else None
        )
        pca_projection = None
        random_spikes = None
        if principal_components_extension is not None:
            try:
                pca_projection = np.asarray(principal_components_extension.get_data())
                random_spikes_extension = (
                    analyzer.get_extension("random_spikes")
                    if analyzer.has_extension("random_spikes")
                    else None
                )
                if random_spikes_extension is not None:
                    random_spikes = random_spikes_extension.get_random_spikes()
                if (
                    pca_projection.ndim not in (2, 3)
                    or pca_projection.shape[0] != len(random_spikes)
                    or pca_projection.shape[1] < 2
                ):
                    raise ValueError("principal_components 数据维度不足以绘制二维 PCA")
            except Exception as exc:
                print(f"读取 principal_components 扩展失败，跳过 group PCA 图：{exc}")
                pca_projection = None
                random_spikes = None

        sampling_frequency = float(analyzer.sampling_frequency)

        def select_evenly(waveforms, max_count):
            """
            在全部 waveforms 中均匀抽样，避免只展示记录开始处的 spikes。

            Parameters
            ----------
            waveforms : np.ndarray
                形状为 ``(波形数, 时间点数)`` 的数组。
            max_count : int
                最多保留的波形数量。

            Returns
            -------
            np.ndarray
                均匀抽样后的波形；数量不足时返回原数组。
            """
            if waveforms.shape[0] <= max_count:
                return waveforms
            indices = np.linspace(0, waveforms.shape[0] - 1, max_count, dtype=int)
            return waveforms[indices]

        def get_peak_channel_waveforms_uv(unit_id):
            """
            获取峰峰值最大通道的全部波形，并确保数值单位为微伏。

            Parameters
            ----------
            unit_id : int | str
                需要提取波形的 Unit ID。

            Returns
            -------
            waveforms_uv : np.ndarray
                形状为 ``(波形数, 时间点数)`` 的微伏波形。
            median_waveform_uv : np.ndarray
                由全部波形计算的中位波形。
            peak_channel_id : int | str
                峰峰值最大的通道 ID。
            time_ms : np.ndarray
                相对于 spike 对齐点的时间轴，单位为毫秒。

            Raises
            ------
            ValueError
                Unit 没有波形，或原始 ADC 值无法换算为微伏时抛出。
            """
            unit_waveforms = np.asarray(
                waveform_extension.get_waveforms_one_unit(unit_id, force_dense=False)
            )
            if unit_waveforms.ndim != 3 or unit_waveforms.shape[0] == 0:
                raise ValueError(f"Unit {unit_id} 没有可用于绘图的 waveforms")

            if analyzer.sparsity is None:
                channel_indices = np.arange(len(analyzer.channel_ids))
            else:
                channel_indices = analyzer.sparsity.unit_id_to_channel_indices[unit_id]

            # 中位数比均值更不容易受到少数异常波形影响。
            median_waveforms = np.nanmedian(unit_waveforms, axis=0)
            peak_to_peak = np.ptp(median_waveforms, axis=0)
            local_peak_index = int(np.nanargmax(peak_to_peak))
            recording_channel_index = int(channel_indices[local_peak_index])
            peak_channel_id = analyzer.channel_ids[recording_channel_index]
            waveforms_uv = unit_waveforms[:, :, local_peak_index].astype(
                "float32", copy=False
            )

            # 新 Analyzer 默认已经以 μV 保存 waveforms。兼容旧 Analyzer：若保存的是
            # ADC 值，则使用对应通道的 gain_to_uV 和 offset_to_uV 显式换算。
            if not analyzer.return_in_uV:
                recording = analyzer.recording
                if recording is None or not recording.has_scaleable_traces():
                    raise ValueError(
                        "Analyzer 中的 waveforms 不是 μV，且 Recording 缺少 "
                        "gain_to_uV/offset_to_uV，无法转换为绝对微伏数值"
                    )
                gain_to_uv = recording.get_channel_gains()[recording_channel_index]
                offset_to_uv = recording.get_channel_offsets()[recording_channel_index]
                waveforms_uv = waveforms_uv * gain_to_uv + offset_to_uv

            median_waveform_uv = np.nanmedian(waveforms_uv, axis=0)
            sample_indices = np.arange(waveforms_uv.shape[1]) - waveform_extension.nbefore
            time_ms = sample_indices / sampling_frequency * 1000.0
            return waveforms_uv, median_waveform_uv, peak_channel_id, time_ms

        def plot_waveforms(
            axis,
            time_ms,
            shown_waveforms_uv,
            median_waveform_uv,
            unit_id,
            peak_channel_id,
            total_count,
        ):
            """将单次波形和由全部数据计算的中位波形绘制到指定坐标轴。"""
            for waveform in shown_waveforms_uv:
                axis.plot(
                    time_ms,
                    waveform,
                    color="0.35",
                    alpha=0.14,
                    linewidth=0.55,
                )

            axis.plot(
                time_ms,
                median_waveform_uv,
                color="orangered",
                linewidth=2.2,
                label="Median waveform (all)",
            )

            # PNG 需要在不同 Unit 之间直接比较振幅，因此统一使用固定纵轴。
            # 超出范围的部分会在 PNG 中被裁剪，但 HTML 仍可自动缩放和交互查看。
            axis.set_ylim(-200.0, 200.0)
            axis.plot(
                [0.0, 0.0],
                [-200.0, 200.0],
                color="steelblue",
                linestyle="--",
                linewidth=1.0,
            )

            axis.set_title(
                f"Unit {unit_id} | peak channel {peak_channel_id} | "
                f"shown {shown_waveforms_uv.shape[0]}/{total_count}"
            )
            axis.set_xlabel("Time relative to spike (ms)")
            axis.set_ylabel("Voltage (μV)")
            axis.grid(alpha=0.18)
            axis.legend(loc="best")

        def plot_autocorrelogram(axis, unit_id):
            """将指定 Unit 的自相关计数绘制到指定坐标轴。"""
            unit_index = analyzer.sorting.id_to_index(unit_id)
            autocorrelogram = all_correlograms[unit_index, unit_index]
            bin_widths = np.diff(correlogram_bins_ms)
            axis.bar(
                correlogram_bins_ms[:-1],
                autocorrelogram,
                width=bin_widths * 0.92,
                align="edge",
                color="#2a9d8f",
                alpha=0.9,
                edgecolor="white",
                linewidth=0.25,
            )
            axis.axvline(0.0, color="#264653", linewidth=0.9, alpha=0.8)
            axis.set_title(f"Unit {unit_id} autocorrelogram")
            axis.set_xlabel("Lag (ms)")
            axis.set_ylabel("Spike-pair count")
            axis.grid(axis="y", alpha=0.22, linewidth=0.7)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        def create_waveform_html_figure(
            time_ms,
            shown_waveforms_uv,
            median_waveform_uv,
            unit_id,
            peak_channel_id,
            total_count,
        ):
            """
            创建 WebGL waveform 图，并把多条波形合并成一个 Plotly trace。

            将所有曲线放进一个 trace 能避免浏览器维护数百个 SVG/JavaScript 对象；
            每条曲线之间插入 NaN，因此它们仍会显示为互相独立的波形。
            """
            waveform_count, sample_count = shown_waveforms_uv.shape
            segmented_time = np.full(
                (waveform_count, sample_count + 1),
                np.nan,
                dtype="float32",
            )
            segmented_voltage = np.full_like(segmented_time, np.nan)
            segmented_time[:, :sample_count] = time_ms
            segmented_voltage[:, :sample_count] = shown_waveforms_uv

            figure = go.Figure()
            figure.add_trace(
                go.Scattergl(
                    x=segmented_time.ravel(),
                    y=segmented_voltage.ravel(),
                    mode="lines",
                    line={"color": "rgba(80, 80, 80, 0.16)", "width": 0.7},
                    name="Single waveforms",
                    hoverinfo="skip",
                    connectgaps=False,
                )
            )
            figure.add_trace(
                go.Scattergl(
                    x=time_ms,
                    y=median_waveform_uv,
                    mode="lines",
                    line={"color": "orangered", "width": 3},
                    name="Median waveform (all)",
                )
            )
            figure.add_vline(
                x=0.0,
                line_width=1,
                line_dash="dash",
                line_color="steelblue",
            )
            figure.update_layout(
                title=(
                    f"Unit {unit_id} | peak channel {peak_channel_id} | "
                    f"shown {waveform_count}/{total_count}"
                ),
                xaxis_title="Time relative to spike (ms)",
                yaxis_title="Voltage (μV)",
                template="plotly_white",
                hovermode="x unified",
                width=1000,
                height=600,
            )
            return figure

        def create_autocorrelogram_html_figure(unit_id):
            """创建 Plotly 自相关直方图。"""
            unit_index = analyzer.sorting.id_to_index(unit_id)
            autocorrelogram = all_correlograms[unit_index, unit_index]
            bin_centers_ms = (
                correlogram_bins_ms[:-1] + correlogram_bins_ms[1:]
            ) / 2.0
            figure = go.Figure(
                data=[
                    go.Bar(
                        x=bin_centers_ms,
                        y=autocorrelogram,
                        width=np.diff(correlogram_bins_ms) * 0.92,
                        marker={
                            "color": "#2a9d8f",
                            "line": {"color": "rgba(255,255,255,0.75)", "width": 0.35},
                        },
                        hovertemplate="Lag: %{x:.2f} ms<br>Count: %{y}<extra></extra>",
                    )
                ]
            )
            figure.update_layout(
                title=f"Unit {unit_id} autocorrelogram",
                xaxis_title="Lag (ms)",
                yaxis_title="Spike-pair count",
                template="plotly_white",
                bargap=0.04,
                hovermode="x unified",
                width=800,
                height=500,
                margin={"l": 70, "r": 30, "t": 75, "b": 65},
            )
            figure.add_vline(x=0.0, line_width=1, line_color="#264653")
            return figure

        def create_group_waveform_html_figure(group_name, group_waveforms):
            """将同一 group 内各 Unit 的单次波形绘制到同一 Plotly 图。

            每个 Unit 的波形合并成一个 Scattergl trace，并用 NaN 分隔不同波形；
            这样既能展示单次波形，又不会为每条曲线创建一个浏览器对象。
            """
            colors = [
                "#264653", "#e76f51", "#2a9d8f", "#f4a261", "#457b9d",
                "#6a4c93", "#d62828", "#2b9348", "#ff006e", "#8338ec",
            ]
            figure = go.Figure()
            for index, (unit_id, time_ms, shown_waveforms_uv, median_waveform_uv, peak_channel_id) in enumerate(
                group_waveforms
            ):
                color = colors[index % len(colors)]
                waveform_count, sample_count = shown_waveforms_uv.shape
                segmented_time = np.full(
                    (waveform_count, sample_count + 1), np.nan, dtype="float32"
                )
                segmented_voltage = np.full_like(segmented_time, np.nan)
                segmented_time[:, :sample_count] = time_ms
                segmented_voltage[:, :sample_count] = shown_waveforms_uv
                figure.add_trace(
                    go.Scattergl(
                        x=segmented_time.ravel(),
                        y=segmented_voltage.ravel(),
                        mode="lines",
                        name=f"{unit_id} waveforms",
                        line={"color": color, "width": 0.75},
                        opacity=0.32,
                        connectgaps=False,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                figure.add_trace(
                    go.Scattergl(
                        x=time_ms,
                        y=median_waveform_uv,
                        mode="lines",
                        name=str(unit_id),
                        line={"color": color, "width": 2.5},
                        hovertemplate=(
                            f"Unit {unit_id}"
                            "<br>Time: %{x:.3f} ms<br>Voltage: %{y:.2f} μV"
                            "<extra></extra>"
                        ),
                    )
                )
            figure.add_vline(x=0.0, line_width=1, line_dash="dash", line_color="#6c757d")
            figure.update_layout(
                title=f"Group {group_name} peak-channel waveforms",
                xaxis_title="Time relative to spike (ms)",
                yaxis_title="Voltage (μV)",
                template="plotly_white",
                hovermode="x unified",
                width=1000,
                height=650,
                margin={"l": 70, "r": 30, "t": 75, "b": 65},
                legend={"title": {"text": "Unit ID"}},
            )
            return figure

        def get_unit_pca_points(unit_id):
            """获取一个 Unit 的二维 PCA 投影，并与随机波形一一对应。"""
            if pca_projection is None or random_spikes is None:
                return None
            unit_index = analyzer.sorting.id_to_index(unit_id)
            unit_mask = random_spikes["unit_index"] == unit_index
            projection = pca_projection[unit_mask]
            if projection.shape[0] == 0:
                return None
            if projection.ndim == 2:
                return projection[:, :2].astype("float32", copy=False)

            # by_channel_local/by_channel_global 的投影还包含通道维度。
            # 优先选该 Unit 的 peak channel；取不到时才对通道维度求均值。
            local_channel_index = None
            try:
                if analyzer.sparsity is None:
                    peak_channel_index = int(
                        np.flatnonzero(analyzer.channel_ids ==
                                       get_peak_channel_waveforms_uv(unit_id)[2])[0]
                    )
                    local_channel_index = peak_channel_index
                else:
                    channel_indices = np.asarray(
                        analyzer.sparsity.unit_id_to_channel_indices[unit_id]
                    )
                    peak_channel_id = get_peak_channel_waveforms_uv(unit_id)[2]
                    peak_global_index = int(
                        np.flatnonzero(analyzer.channel_ids == peak_channel_id)[0]
                    )
                    matches = np.flatnonzero(channel_indices == peak_global_index)
                    if matches.size:
                        local_channel_index = int(matches[0])
            except Exception:
                local_channel_index = None

            if (
                local_channel_index is not None
                and local_channel_index < projection.shape[2]
            ):
                return projection[:, :2, local_channel_index].astype(
                    "float32", copy=False
                )
            return np.nanmean(projection[:, :2, :], axis=2).astype(
                "float32", copy=False
            )

        def create_group_pca_html_figure(group_name, group_pca_points):
            """绘制同一 group 内各 Unit 的二维 PCA 投影。"""
            colors = [
                "#264653", "#e76f51", "#2a9d8f", "#f4a261", "#457b9d",
                "#6a4c93", "#d62828", "#2b9348", "#ff006e", "#8338ec",
            ]
            figure = go.Figure()
            for index, (unit_id, points) in enumerate(group_pca_points):
                color = colors[index % len(colors)]
                figure.add_trace(
                    go.Scattergl(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode="markers",
                        name=str(unit_id),
                        marker={"color": color, "size": 5, "opacity": 0.65},
                        hovertemplate=(
                            f"Unit {unit_id}"
                            "<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>"
                        ),
                    )
                )
            figure.update_layout(
                title=f"Group {group_name} two-dimensional PCA",
                xaxis_title="PC1",
                yaxis_title="PC2",
                template="plotly_white",
                hovermode="closest",
                width=900,
                height=700,
                margin={"l": 70, "r": 30, "t": 75, "b": 65},
                legend={"title": {"text": "Unit ID"}},
            )
            return figure

        print("开始生成可视化图表...")
        group_waveforms = {}
        group_pca_points = {}
        unit_groups = None
        if plot_group_waveforms and waveform_extension is not None:
            try:
                unit_groups = analyzer.sorting.get_property("group")
                if unit_groups is None or len(unit_groups) != len(analyzer.unit_ids):
                    print("Sorting 缺少与 Unit 对应的 group 属性，跳过 group 波形图。")
                    unit_groups = None
            except Exception as exc:
                print(f"读取 Unit group 属性失败，跳过 group 波形图：{exc}")

        for unit_index, unit_id in enumerate(analyzer.unit_ids):
            print(f"  处理 Unit {unit_id}...")
            waveforms_uv = None
            median_waveform_uv = None
            peak_channel_id = None
            time_ms = None
            if waveform_extension is not None:
                try:
                    waveforms_uv, median_waveform_uv, peak_channel_id, time_ms = (
                        get_peak_channel_waveforms_uv(unit_id)
                    )
                except Exception as exc:
                    print(f"    Unit {unit_id} 的 waveform 图跳过：{exc}")

            if (
                plot_group_waveforms
                and unit_groups is not None
                and median_waveform_uv is not None
            ):
                group_name = str(unit_groups[unit_index])
                group_waveforms.setdefault(group_name, []).append(
                    (
                        unit_id,
                        time_ms,
                        select_evenly(waveforms_uv, max_waveforms_group),
                        median_waveform_uv,
                        peak_channel_id,
                    )
                )
                if pca_projection is not None and random_spikes is not None:
                    try:
                        pca_points = get_unit_pca_points(unit_id)
                        if pca_points is not None:
                            group_pca_points.setdefault(group_name, []).append(
                                (unit_id, select_evenly(pca_points, max_waveforms_group))
                            )
                    except Exception as exc:
                        print(f"    Unit {unit_id} 的 PCA 投影跳过：{exc}")

            total_count = analyzer.sorting.get_unit_spike_train(unit_id=unit_id).shape[0]

            # 清理旧版本可能遗留、但新版本不再需要的图表。
            for stale_path in (
                html_folder / f"{unit_id}_density.html",
                html_folder / f"{unit_id}_amplitudes.html",
            ):
                if stale_path.exists():
                    stale_path.unlink()

            # 1. waveform HTML：只有计算了 waveforms 扩展时才生成。
            if waveforms_uv is not None:
                html_waveforms = select_evenly(waveforms_uv, max_waveforms_html)
                png_waveforms = select_evenly(waveforms_uv, max_waveforms_png)
                waveform_html_figure = create_waveform_html_figure(
                    time_ms,
                    html_waveforms,
                    median_waveform_uv,
                    unit_id,
                    peak_channel_id,
                    total_count,
                )
                waveform_html_figure.write_html(
                    html_folder / f"{unit_id}_waveforms.html",
                    include_plotlyjs="cdn",
                    full_html=True,
                    config={"scrollZoom": True, "displaylogo": False},
                )

            # 2. autocorrelogram HTML：只有计算了 correlograms 扩展时才生成。
            autocorr_available = False
            if correlogram_extension is not None:
                try:
                    autocorr_html_figure = create_autocorrelogram_html_figure(unit_id)
                    autocorr_html_figure.write_html(
                        html_folder / f"{unit_id}_autocorr.html",
                        include_plotlyjs="cdn",
                        full_html=True,
                        config={"scrollZoom": True, "displaylogo": False},
                    )
                    autocorr_available = True
                except Exception as exc:
                    print(f"    Unit {unit_id} 的自相关图跳过：{exc}")

            # 3. PNG 只绘制已经计算出的子图；两个扩展都缺失时不生成 PNG。
            available_plots = int(waveforms_uv is not None) + int(autocorr_available)
            if available_plots > 0:
                combined_figure, axes = plt.subplots(
                    1,
                    available_plots,
                    figsize=(7.5 * available_plots, 5.5),
                    squeeze=False,
                )
                axes = axes[0]
                axis_index = 0
                try:
                    if waveforms_uv is not None:
                        plot_waveforms(
                            axes[axis_index],
                            time_ms,
                            png_waveforms,
                            median_waveform_uv,
                            unit_id,
                            peak_channel_id,
                            total_count,
                        )
                        axis_index += 1
                    if autocorr_available:
                        try:
                            plot_autocorrelogram(axes[axis_index], unit_id)
                        except Exception as exc:
                            print(f"    Unit {unit_id} 的 PNG 自相关子图跳过：{exc}")
                    combined_figure.subplots_adjust(
                        left=0.07,
                        right=0.98,
                        bottom=0.13,
                        top=0.89,
                        wspace=0.25,
                    )
                    combined_figure.savefig(
                        fig_folder / f"{unit_id}_abstract.png",
                        dpi=300,
                    )
                finally:
                    plt.close(combined_figure)

        # group 汇总图展示每个 Unit 抽样后的单次波形及中位波形，不添加分类参数框。
        if group_waveforms:
            for group_name, group_data in group_waveforms.items():
                safe_group_name = "".join(
                    char if char.isalnum() or char in ("-", "_") else "_"
                    for char in group_name
                )
                group_figure = create_group_waveform_html_figure(group_name, group_data)
                group_figure.write_html(
                    html_folder / f"group_{safe_group_name}_waveforms.html",
                    include_plotlyjs="cdn",
                    full_html=True,
                    config={"scrollZoom": True, "displaylogo": False},
                )

        if group_pca_points:
            for group_name, group_data in group_pca_points.items():
                safe_group_name = "".join(
                    char if char.isalnum() or char in ("-", "_") else "_"
                    for char in group_name
                )
                pca_figure = create_group_pca_html_figure(group_name, group_data)
                pca_figure.write_html(
                    html_folder / f"group_{safe_group_name}_pca.html",
                    include_plotlyjs="cdn",
                    full_html=True,
                    config={"scrollZoom": True, "displaylogo": False},
                )


class _ExportMixin:
    """按职责组织的 Export 操作。重型依赖保持方法内懒加载。"""
    def export_to_phy(self, analyzer_folder, output_folder):
        """
        导出到Phy格式
        # 导出到phy，供cellexplorer打开。注意cellexplorer所读取的recording是int16格式的原始文件，并且phy的默认LSB是0.195µV/bit
        # 易格的设备输出的LSB不是这个值
        # 如果要用phy打开，需要使用powershell管理员，conda activate到phy环境，再输入phy template-gui  folder_phy\\params.py
        Args:
            analyzer_folder: 分析器文件夹
            output_folder: 输出Phy文件夹
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import spikeinterface.exporters as ex
        import yyl_utils as yyl

        yyl.check_delete_exists_path(output_folder)

        analyzer = si.load(analyzer_folder)
        ex.export_to_phy(
            sorting_analyzer=analyzer,
            output_folder=output_folder,
            remove_if_exists=True
        )

    def run_cell_explorer(self, phy_folder, matlab_function_path, output_metrics_path):
        """
        运行CellExplorer分析

        Args:
            phy_folder: Phy数据文件夹
            matlab_function_path: MATLAB函数路径
            output_metrics_path: 输出指标文件路径
        """
        # 检查 MATLAB 是否可用（使用实例变量）
        if not self._matlab_available:
            print("MATLAB引擎不可用，跳过CellExplorer分析")
            return None

        # 延迟导入 matlab（在确认可用之后）
        import matlab.engine
        import traceback
        import yyl_utils as yyl
        import os
        import joblib
        import pandas as pd
        from pathlib import Path

        # CellExplorer 只读取 cluster_group.tsv 中 group=good 的 Unit。
        # 因此在调用 MATLAB 管道前，将 Phy 导出的所有 Unit 标记为 good。
        def label_all_good(tsv_path):
            """将 Phy 中的所有单元标记为 good。"""
            tsv_path = Path(tsv_path)
            if not tsv_path.exists():
                print(f"未找到 {tsv_path}，跳过 Phy Unit good 标记。")
                return
            df = pd.read_csv(tsv_path, sep="\t")
            if "group" not in df.columns:
                print(f"{tsv_path} 中没有 group 列，跳过 Phy Unit good 标记。")
                return
            df["group"] = "good"
            df.to_csv(tsv_path, sep="\t", index=False)

        label_all_good(Path(phy_folder) / "cluster_group.tsv")

        # 内部MATLAB管道类
        class _MatlabPipeline:
            """MATLAB管道内部类"""

            def __init__(self):
                self.eng = None
                self.is_connected = False

            def connect(self):
                """连接到MATLAB引擎"""
                try:
                    print("正在启动MATLAB引擎...")
                    self.eng = matlab.engine.start_matlab()
                    self.is_connected = True
                    print("MATLAB引擎启动成功")
                    return True
                except Exception as e:
                    print(f"启动MATLAB引擎失败: {e}")
                    return False

            def disconnect(self):
                """断开MATLAB连接"""
                if self.eng and self.is_connected:
                    print("关闭MATLAB引擎...")
                    self.eng.quit()
                    self.is_connected = False

            def run_pipeline(self, function_path, basepath):
                """
                运行MATLAB管道

                Parameters:
                function_path (str): MATLAB函数路径
                basepath (str): 基础数据路径

                Returns:
                tuple: (success, result, error_message)
                """
                if not self.is_connected:
                    if not self.connect():
                        return False, None, "无法连接到MATLAB引擎"

                try:
                    if not os.path.exists(function_path):
                        return False, None, f"function_path不存在: {function_path}"

                    print("添加MATLAB函数路径...")
                    self.eng.addpath(self.eng.genpath(function_path))

                    print("调用yyl_pipeline_cellexplorer函数...")
                    cell_metrics = self.eng.yyl_pipeline_cellexplorer(basepath, function_path, nargout=1)

                    return True, cell_metrics, None

                except Exception as e:
                    error_msg = f"MATLAB函数执行错误: {str(e)}"
                    print("=" * 50)
                    print("错误详情:")
                    traceback.print_exc()
                    print("=" * 50)
                    return False, None, error_msg

            def __enter__(self):
                self.connect()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.disconnect()

        # 使用上下文管理器，确保 MATLAB Engine 无论成功还是异常都会关闭。
        with _MatlabPipeline() as pipeline:
            success, result, error_msg = pipeline.run_pipeline(
                str(matlab_function_path),
                str(phy_folder)
            )

        if success:
            print("MATLAB管道执行成功!")
            yyl.check_delete_exists_path(output_metrics_path)
            joblib.dump(result, output_metrics_path, compress=3)
            return result
        else:
            print(f"MATLAB管道执行失败: {error_msg}")
            return None


class _PipelineRunnerMixin:
    """按职责组织的 PipelineRunner 操作。重型依赖保持方法内懒加载。"""
    def run_pipeline(
        self,
        probe: Probe = None,
        config=None,
        traces_config=None,
        n_jobs=-1,
        run_until_step=7,
    ):
        """
        按顺序运行 spike sorting 全流程，并可在指定步骤完成后停止。

        Parameters
        ----------
        probe : Probe | None, default: None
            探针结构。可以通过 ``SpikeSortingPipeline.get_probe()`` 创建。
        config : dict | None, default: None
            管道配置。支持以下键（未提供时使用相应函数的默认值）：

            - ``sorter_name``：sorter 名称，默认 ``"mountainsort5"``。
            - ``sorting_params``：sorter 参数；未提供时读取该 sorter 的默认值。
            - ``extensions_dict``：Analyzer 扩展参数；None 使用本项目默认配置。
            - ``preprocess_pipeline_dict``：预处理 Pipeline；None 使用项目默认流程，
              空字典表示跳过所有预处理操作。
            - ``sorting_in_docker``：是否使用docker进行sorting。注意python需安装docker包，
              且docker需打开。
            - ``max_waveforms_html``：每个 Unit 的 HTML 最多绘制多少条波形，
              默认 2000。
            - ``max_waveforms_png``：每个 Unit 的 PNG 最多绘制多少条波形，
              默认 2000。
            - ``max_waveforms_group``：每个 Unit 在 group waveform 和 group PCA
              HTML 中最多绘制多少条波形/投影点，默认 2000。
            ``sorting_params``、``extensions_dict`` 和 ``preprocess_pipeline_dict``
            可分别通过 ``get_all_sorting_params_dict()``、
            ``get_all_extensions_params_dict()`` 和
            ``get_all_preprocess_pipeline_dict()`` 查询可用参数和默认值。
        traces_config : dict | None, default: None
            从 NumPy 数组创建 Recording 时使用。传入该字典后，不再从 PLX 文件读取。
            字典中的字段如下：

            - ``traces`` (np.ndarray)：神经信号数据，形状为
              ``(n_samples, n_channels)``；默认单位为 μV，该单位会影响后续阈值、
              去噪和波形绘图。
            - ``fs`` (float)：采样频率，单位 Hz。
            - ``chan_ids`` (list)：通道 ID 列表，长度必须等于 ``n_channels``。
            - ``metadata_folder`` (str | Path | None)：可选的元数据文件夹，包含
              通道增益等 Recording 元数据。
            - ``properties`` (dict | None)：可选的 Recording 属性字典，例如 LSB
              等信号处理属性。
            如果 ``traces_config`` 为 None，则从 ``self.input_file`` 指向的 PLX 文件读取。
        n_jobs : int, default: -1
            支持并行的步骤所使用的任务数；-1 表示使用全部可用核心。
        run_until_step : int | str | None, default: 7
            控制管道运行到哪个步骤后停止。可以传 1～8，也可以传下面列出的
            英文步骤名。默认运行到步骤 7（导出 Phy），不会自动启动 CellExplorer；
            传入 8 或 ``"cell_explorer"`` 才会继续执行 CellExplorer。

        Returns
        -------
        None

        Workflow
        --------
        1. ``save_recording``：从 PLX 或 NumPy 数组创建并保存原始 Recording。
        2. ``preprocess``：应用预处理 Pipeline 并保存结果。
        3. ``sorting``：运行 sorter 并保存 Sorting。
        4. ``create_analyzer``：创建 Analyzer、计算 extensions、导出指标表格。
        5. ``renew_unit_type``：质量筛选并可选推断细胞类型。
        6. ``visualize``：生成 waveform、自相关和 group 汇总 HTML/PNG。
        7. ``export_to_phy``：导出 Phy 文件。
        8. ``cell_explorer``：在 ``run_until_step=8`` 且 MATLAB Engine 可用时运行。

        Notes
        -----
        ``run_until_step`` 只控制“在哪里停止”，不会跳过前面的步骤。例如传入
        4 会依次完成步骤 1～4，然后返回。CellExplorer 还需要 MATLAB Engine 可用。
        """
        if config is None:
            config = {}
        if traces_config is None:
            traces_config = {}

        step_names = {
            1: "save_recording",
            2: "preprocess",
            3: "sorting",
            4: "create_analyzer",
            5: "renew_unit_type",
            6: "visualize",
            7: "export_to_phy",
            8: "cell_explorer",
        }
        step_numbers = {name: number for number, name in step_names.items()}

        # run_until_step 只从函数参数传入，不再从 config 读取。
        requested_step = 7 if run_until_step is None else run_until_step

        if isinstance(requested_step, str):
            normalized_step = requested_step.strip().lower()
            if normalized_step not in step_numbers:
                raise ValueError(
                    "run_until_step 必须是 1～8，或以下步骤名之一："
                    f"{list(step_numbers)}"
                )
            final_step = step_numbers[normalized_step]
        elif isinstance(requested_step, int) and not isinstance(requested_step, bool):
            if requested_step not in step_names:
                raise ValueError("run_until_step 必须是 1～8 之间的整数")
            final_step = requested_step
        else:
            raise TypeError("run_until_step 必须是 int、str 或 None")

        def stop_after(step_number):
            """完成用户指定的最后一步后打印提示并返回停止标记。"""
            if final_step == step_number:
                print(
                    f"已完成步骤 {step_number}/8 "
                    f"({step_names[step_number]})，按 run_until_step 设置停止。"
                )
                return True
            return False

        # 集中读取配置，使下方每个步骤只保留实际执行逻辑。
        sorter_name = config.get("sorter_name", "mountainsort5")
        sorting_params = config.get("sorting_params")
        if sorting_params is None:
            sorting_params = self._get_current_sorting_params_dict(sorter_name)
        extensions_dict = config.get("extensions_dict")
        preprocess_pipeline_dict = config.get("preprocess_pipeline_dict")
        run_in_docker = config.get("run_in_docker")
        max_waveforms_html = config.get("max_waveforms_html", 2000)
        max_waveforms_png = config.get("max_waveforms_png", 2000)
        max_waveforms_group = config.get("max_waveforms_group", 2000)
        # run_pipeline 固定一次只运行一个 group sorter。Kilosort4 通常使用 GPU，
        # 固定为 1 可以避免多个任务同时争抢显存。直接调用 perform_sorting() 时，
        # 仍可通过其 group_n_jobs 参数自行控制 joblib 并行数。
        group_n_jobs = 1

        print(f"本次管道计划运行到步骤 {final_step}/8 ({step_names[final_step]})。")

        print("步骤 1/8：创建并保存原始 Recording...")
        if traces_config:
            self.save_traces_to_recording_file(
                traces_config.get("traces"),
                traces_config.get("fs"),
                traces_config.get("chan_ids"),
                self.output_paths.raw_recording_folder,
                traces_config.get("metadata_folder"),
                traces_config.get("properties"),
                probe=probe,
                n_jobs=1,
            )
        else:
            self.read_save_plx_file(
                self.input_file,
                self.output_paths.raw_recording_folder,
                probe=probe,
                n_jobs=n_jobs,
            )
        if stop_after(1):
            return

        print("步骤 2/8：应用预处理 Pipeline...")
        self.preprocess_recording(
            self.output_paths.raw_recording_folder,
            self.output_paths.preprocessed_recording_folder,
            n_jobs=n_jobs,
            preprocess_pipeline_dict=preprocess_pipeline_dict,
        )
        if stop_after(2):
            return

        print("步骤 3/8：执行 spike sorting...")
        try:
            sorting = self.perform_sorting(
                self.output_paths.preprocessed_recording_folder,
                sorter_name,
                self.output_paths.sorting_object_folder,
                sorting_params,
                run_in_docker=run_in_docker,
                group_n_jobs=group_n_jobs,
            )
        except Exception as exc:
            print(f"步骤 3 sorting 失败：{exc}")
            print("当前 setting 已停止，跳过 Analyzer、扩展计算和后续步骤。")
            return

        # 某些 group 可能没有任何 spike。此时 perform_sorting() 会返回空
        # NumpySorting；不要将空 sorting 交给 Analyzer，以免扩展内部对空数组
        # 调用 np.concatenate([]) 而再次报错。
        if sorting is None or len(sorting.unit_ids) == 0:
            print("步骤 3 完成，但没有任何有效 Unit。")
            print("跳过 Analyzer、质量指标、可视化、Phy 和 CellExplorer。")
            return

        if stop_after(3):
            return

        try:
            print("步骤 4/8：创建 SortingAnalyzer 并计算 extensions...")
            self.create_analyzer(
                self.output_paths.preprocessed_recording_folder,
                self.output_paths.sorting_object_folder,
                self.output_paths.sorting_analyzer_folder,
                extensions_dict=extensions_dict,
                compute=True,
                template_metrics_path=self.output_paths.template_metrics_path,
                qm_path=self.output_paths.qm_excel_path,
                n_jobs=n_jobs,
            )
            if stop_after(4):
                return

            print("步骤 5/8：更新 Unit 质量和细胞类型结果...")
            self.renew_unit_type(
                self.output_paths.sorting_analyzer_folder,
                self.output_paths.cell_type_metrics_path,
            )
            if stop_after(5):
                return

            print("步骤 6/8：生成 waveform、自相关和 group 汇总图表...")
            self.visualize_results(
                self.output_paths.sorting_analyzer_folder,
                fig_folder=self.output_paths.figures_folder / "waveform_figures",
                max_waveforms_html=max_waveforms_html,
                max_waveforms_png=max_waveforms_png,
                max_waveforms_group=max_waveforms_group,
            )
            if stop_after(6):
                return

            print("步骤 7/8：导出 Phy 文件...")
            self.export_to_phy(
                self.output_paths.sorting_analyzer_folder,
                self.output_paths.phy_folder,
            )
            if stop_after(7):
                return

            print("步骤 8/8：检查并运行 CellExplorer...")
            if not self._matlab_available:
                print("MATLAB Engine 不可用，跳过 CellExplorer。")
            else:
                self.run_cell_explorer(
                    self.output_paths.phy_folder,
                    self.matlab_func_path,
                    self.output_paths.cell_metrics_path,
                )

            print("管道执行完成！")

        except ValueError as exc:
            if "need at least one array to concatenate" in str(exc):
                print(exc)
                print("Sorting 中存在空 Unit，已停止当前文件的后续分析。")
                return
            raise


class SpikeSortingPipeline(_ParameterCatalogMixin, _RecordingMixin, _SortingMixin, _AnalyzerMixin, _UnitAnalysisMixin, _VisualizationMixin, _ExportMixin, _PipelineRunnerMixin):
    """尖峰排序处理管道类。

    具体操作按职责拆分到多个内部组件类；本类保留原有公开接口，
    通过多个组件的继承组合执行原来的完整流程。各组件中的重型
    依赖仍在方法内部导入，不会在模块加载时提前导入。
    """
    _OutputPaths = _PipelineOutputPaths

    def __init__(self, input_file: str = r"", output_folder: str = r""):
        """
        初始化管道

        Args:
            input_file: 输入plx文件
            output_folder: 输出结果的总目录，目录下包括recording文件夹、sorting文件夹、figure文件夹之类
            n_jobs: 并行工作数
        """

        # 只导入必要的轻量级包
        from pathlib import Path

        self.input_file = Path(input_file)
        self.matlab_func_path = r""
        self.curation_model_path = r""
        self.output_paths: SpikeSortingPipeline._OutputPaths = self._OutputPaths(Path(output_folder))

        # 检查 MATLAB 是否可用（不实际导入，只做可用性检查）
        self._matlab_available = False
        try:
            import matlab.engine
            self._matlab_available = True
        except ImportError:
            self._matlab_available = False