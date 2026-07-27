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
    & conda activate phy2
    & phy template-gui "{params_path}"
    '''
    subprocess.run(['powershell', '-Command', cmd])

class SpikeSortingPipeline:
    """尖峰排序处理管道类"""

    class _OutputPaths:
        def __init__(self, base_folder: Path):
            self.base_folder: Path = base_folder
            self.raw_recording_folder = base_folder / 'raw_recording_folder'
            self.preprocessed_recording_folder = base_folder / 'preprocessed_recording_folder'
            self.whittened_recording_folder = base_folder / 'whittened_recording_folder'
            self.sorting_verbose_folder = base_folder / 'sorting_verbose_folder'
            self.sorting_object_folder = base_folder / 'sorting_object_folder'
            self.sorting_analyzer_folder = base_folder / 'sorting_analyzer_folder'
            self.phy_folder = base_folder / 'phy_folder'
            self.figures_folder = base_folder / 'figures_folder'
            self.qm_excel_path = base_folder / 'qm_excel.xlsx'
            self.template_metrics_path = base_folder / 'template_metrics.xlsx'
            self.cell_metrics_path = base_folder / "cell_metrics.joblib"
            self.cell_type_metrics_path = base_folder / "cell_type_metrics.xlsx"

    def __init__(self, input_file: str = r"", output_folder: str = r""):
        """
        初始化管道

        Args:
            input_file: 输入plx文件
            output_folder: 输出结果的总目录，目录下包括recording文件夹、sorting文件夹、figure文件夹之类
            n_jobs: 并行工作数
        """

        # 只导入必要的轻量级包
        import os
        os.environ['KACHERY_API_KEY'] = 'iGnRNcwk2uPk552dRakTwLScoUS78DIU'

        import yyl_utils as yyl
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path
        import joblib
        import traceback

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

    @staticmethod
    def get_default_sorting_params(sorter_name):
        """获取默认排序参数"""
        if sorter_name.lower() == 'mountainsort5':
            return {
                "scheme": "2",
                "detect_threshold": 5,
                "detect_sign": -1,
                "detect_time_radius_msec": 0.25,
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
                "freq_min": 300,
                "freq_max": 6000,
                "filter": False,
                "whiten": True,
                "chunk_duration": "1s",
                "progress_bar": True,
            }
        else:
            return {}

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
    def get_probe(num_channels: int = 16, positions=None):
        """
        创建探针配置,默认创建的是16通道间隔300um的微丝电极。
        Args:
            num_channels: 通道数量，默认为16通道
            positions: 电极位置列表，每个元素为(x,y)坐标元组。如果为None则自动生成线性排列位置
        Returns:
            probe: 配置好的Probe对象，包含电极几何信息和通道映射
        """
        # 只在需要时导入 probeinterface
        from probeinterface import Probe
        import numpy as np

        n = num_channels
        # 如果未提供位置信息，生成默认的线性排列电极位置（间距300um）
        if positions is None:
            positions = [(i * 300, 0) for i in range(n)]
        # 创建2维探针对象，单位设置为微米(um)
        probe = Probe(ndim=2, si_units='um')
        # 设置电极接触点：圆形电极，半径12.5um
        probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 12.5})
        # 设置设备通道索引：顺序映射0到n-1
        channel_indices = np.arange(n)
        probe.set_device_channel_indices(channel_indices)
        return probe

    def save_traces_to_recording_file(self, traces, fs, chan_ids, outputpath,
                                      metadata_folder=None, properties=None,
                                      probe: Probe = None, n_jobs=1):
        """
        保存记录数据到二进制文件

        Args:
            traces: 神经信号数据，形状为 [时间点, 通道数] 的numpy数组
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

        # 给重建的recording加上探针信息
        if probe:
            # 如果提供了探针对象，使用该探针
            rec_fixed.set_probe(probe, in_place=True)
        else:
            # 如果没有提供探针，创建默认的线性探针
            rec_fixed.set_probe(self.get_probe(num_channels=len(chan_ids)), in_place=True)

        if metadata_folder:
            # 给重建的recording复制metadata（如通道信息、增益等）
            rec_fixed.load_metadata_from_folder(metadata_folder)

        if properties:
            # 给重建的recording加上LSB等属性，_properties中包括LSB信息
            rec_fixed._properties = properties

        # 保存重建的recording到二进制文件
        yyl.check_delete_exists_path(outputpath)  # 检查并删除已存在的路径
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
        traces = test_recording.get_traces()  # 得到plx文件的原始数据(未缩放,缩放会导致cellexplorer读取失败)
        fs = test_recording.get_sampling_frequency()  # 得到plx文件的采样率
        chan_ids = test_recording.channel_ids  # 得到plx文件的通道id

        rec_fixed = si.NumpyRecording(
            traces_list=[traces],
            sampling_frequency=fs,
            channel_ids=chan_ids)

        # 给重建的recording加上探针信息
        if probe:
            rec_fixed.set_probe(probe, in_place=True)
        # 给重建的recording复制metadata
        rec_fixed.copy_metadata(test_recording)
        # 给重建的recording加上LSB,_properties中包括LSB信息
        rec_fixed._properties = test_recording._properties

        # 保存重建的recording
        yyl.check_delete_exists_path(outputpath)
        rec_fixed.save(folder=outputpath, format="binary", name='plx测试', verbose=True, n_jobs=n_jobs)
        return rec_fixed

    def preprocess_recording(self, input_folder, output_folder_preprocessed, output_folder_whitened=None, n_jobs=1):
        """
        对原始recording进行预处理

        Args:
            input_folder: 输入的recording文件夹
            output_folder_preprocessed: 进行了其它预处理，白化前的数据
            output_folder_whitened: 白化后的数据(幅值会改变)
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import spikeinterface.preprocessing as spre
        import yyl_utils as yyl

        lsb = 1
        yyl.check_delete_exists_path([output_folder_preprocessed, ])
        raw_recording = si.load(file_or_folder_or_dict=input_folder)
        # 带通滤波
        highpass_recording = spre.filter(recording=raw_recording)
        # 去除基线漂移
        center_recording = spre.center(recording=highpass_recording)
        # 阈值降噪
        threshold_recording = spre.blank_saturation(
            recording=center_recording,
            abs_threshold=0.5 / lsb,
            direction='both'
        )
        # 公共参考
        referenced_recording = spre.common_reference(
            recording=threshold_recording,
            reference='local',
            local_radius=(300, 1000)
        )

        # 保存预处理后的数据
        yyl.check_delete_exists_path(output_folder_preprocessed)
        referenced_recording.save(
            folder=output_folder_preprocessed,
            format="binary",
            verbose=True,
            n_jobs=n_jobs
        )

        # 白化处理（可选）
        if output_folder_whitened:
            whiten_recording = spre.whiten(recording=referenced_recording, dtype='float32')
            yyl.check_delete_exists_path(output_folder_whitened)
            whiten_recording.save(
                folder=output_folder_whitened,
                format="binary",
                verbose=True,
                n_jobs=n_jobs
            )
            return referenced_recording, whiten_recording

        return referenced_recording

    def perform_sorting(self, input_folder, sorter_name, output_folder, params=None):
        """
        执行排序

        Args:
            input_folder: 输入数据文件夹
            sorter_name: 排序器名称
            output_folder: 输出文件夹
            params: 排序参数
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import spikeinterface.sorters as ss
        import yyl_utils as yyl

        # 加载recording
        recording = si.load(input_folder)

        # 设置默认参数
        if params is None:
            params = self.get_default_sorting_params(sorter_name)

        # 运行排序
        yyl.check_delete_exists_path(output_folder)
        job_dict = {
            'sorter_name': sorter_name,
            'recording': recording,
            'folder': output_folder,
            'verbose': True,
            'raise_error': False,
            'remove_existing_folder': True,
            'delete_output_folder': False,
            # , 'docker_image': True, 'delete_container_files': False
            # , 'installation_mode': "folder", 'spikeinterface_folder_source':  # spikeinterface安装包路径
            #  r"C:\Users\32707\Desktop\工作\用户实验\张刘馨黛-数据分析\测试代码\安装包\spikeinterface-main\spikeinterface-main"
        }
        job_dict.update(params)
        sorting = ss.run_sorter(**job_dict)
        return sorting

    def batch_sorting(self, input_folder_list: list[str], output_base_folder, sorter_name, params=None, n_jobs=1):
        """
        批量排序
        注意保存的verbose_path路径不能在同一个子文件夹下，必须是不同的倒数第二级文件夹，不然它们的元文件会串起来进而报错
        Args:
            input_folder_list: 总文件夹，要求文件夹下是一群白化后的recording的数据
            output_base_folder: 输出结果文件夹，包括verbose和sorting结果
            sorter_name: 排序器名称
            params: 排序参数
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import spikeinterface.sorters as ss
        import yyl_utils as yyl

        if params is None:
            params = self.get_default_sorting_params(sorter_name)

        job_list = []
        for index, folder_path in enumerate(input_folder_list):
            recording = si.load(folder_path)
            verbose_path = output_base_folder / f"{index}" / f'verbose{index}'
            yyl.check_delete_exists_path([verbose_path])

            job_dict = {
                'sorter_name': sorter_name,
                'recording': recording.clone(),
                'folder': verbose_path,
                'verbose': True,
                'remove_existing_folder': False,
                'raise_error': False,
                'docker_image': False,
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
            engine_kwargs={'n_jobs': n_jobs}
        )
        return sortings

    def create_analyzer(self, recording_folder, sorting_folder, output_folder, output_qm_path: None, n_jobs=1):
        """
        创建排序分析器

        Args:
            recording_folder: 记录数据文件夹
            sorting_folder: 排序结果文件夹
            output_folder: 输出分析器文件夹
            output_qm_path: 是否保存qm矩阵
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import yyl_utils as yyl

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
            folder=output_folder
        )

        analyzer.compute('random_spikes', n_jobs=n_jobs)
        analyzer.compute('waveforms', n_jobs=n_jobs, ms_before=0.7, ms_after=1.3)
        analyzer.compute('templates', n_jobs=n_jobs)
        analyzer.compute("noise_levels", n_jobs=n_jobs)
        analyzer.compute("amplitude_scalings", n_jobs=n_jobs)
        analyzer.compute("correlograms", n_jobs=n_jobs)
        analyzer.compute("isi_histograms", n_jobs=n_jobs)
        analyzer.compute("principal_components", n_jobs=n_jobs)
        analyzer.compute("spike_amplitudes", n_jobs=n_jobs)
        analyzer.compute("spike_locations", n_jobs=n_jobs)
        analyzer.compute("template_metrics", n_jobs=n_jobs)
        analyzer.compute("template_similarity", n_jobs=n_jobs)
        analyzer.compute("unit_locations", n_jobs=n_jobs)
        analyzer.compute('quality_metrics', n_jobs=n_jobs)

        if output_qm_path:
            (analyzer.extensions['quality_metrics'].data['metrics']).to_excel(self.output_paths.qm_excel_path,
                                                                              index=True)

        ((analyzer.extensions['template_metrics'].data['metrics']).
         to_excel(self.output_paths.template_metrics_path, index=True))

        return analyzer

    def renew_unit_type(self, analyzer_folder, cell_type_metrics_path, classify_units=True):
        """
        判断细胞的分类情况及细胞的可能类型，将结果保存至路径self.output_paths.cell_type_metrics_path
        :param classify_units: 是否判断细胞类型
        :return: None
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import numpy as np
        import pandas as pd

        def screen_units(df_qm_metrics):
            """
            根据质量指标筛选符合分析条件的神经元单元
            """
            # 读取质量指标和模板指标数据
            # 初始化筛选字典：包含所有unit_id和空的质量列表
            screen_dict = {"unit_ids": df_qm_metrics.index.tolist(),
                           "sorting_quality": []}

            # 筛选符合条件的unit进行分析
            for unit_id in screen_dict["unit_ids"]:
                # 筛选条件：ISI违例率<0.5 且 放电率>0.5Hz 且 放电率<50Hz
                if ((df_qm_metrics.loc[unit_id, 'isi_violations_ratio'] < 0.5) &
                        (df_qm_metrics.loc[unit_id, 'firing_rate'] > 0.5) &
                        (df_qm_metrics.loc[unit_id, 'firing_rate'] < 50)
                ):
                    # 符合所有筛选条件的单元标记为"good"
                    screen_dict["sorting_quality"].append("good")
                else:
                    screen_dict["sorting_quality"].append("bad")
            return screen_dict

        def get_unit_max_channel(sorting_analyzer):
            """获取所有unit的最大的channel位置"""
            # 获取所有单元的ID
            unit_ids = sorting_analyzer.unit_ids
            # 获取探针上所有触点的物理坐标位置
            positions = sorting_analyzer.get_probe().contact_positions
            # 获取每个单元的物理位置坐标，只取与触点坐标维度相同的列
            unit_locations = sorting_analyzer.extensions["unit_locations"].get_data()[:, 0:positions.shape[-1]]
            # 获取所有通道的ID
            channel_ids = sorting_analyzer.channel_ids
            # 初始化存储每个单元对应最大通道的列表
            unit_channel_ids = []
            # 遍历每个单元的位置
            for unit_location in unit_locations:
                # 计算当前单元位置到每个触点的欧氏距离
                distances = [np.sqrt(np.sum((unit_location - position) ** 2)) for position in positions]
                # 找到距离最近的触点索引（即最大通道对应的触点）
                channel_idx = np.argmin(distances)
                # 根据索引获取对应的通道ID并添加到列表中
                unit_channel_ids.append(channel_ids[channel_idx])
            # 创建包含单元ID和对应最大通道的字典
            channel_dict = {"unit_ids": unit_ids, "max_channel": unit_channel_ids}

            return channel_dict

        def get_units_classified(df_qm_metrics, df_template_metrics):
            """
            根据波形特征和放电率对神经元进行分类
            分类标准（基于文献报道的锥体神经元和中间神经元特征）：
            -  putative_pyramidal_units（ putative锥体神经元）:
                * peak_to_valley > 0.4 ms（宽波形）
                * 放电率 < 5 Hz（低频放电）

            -  putative_interneuron_units（ putative中间神经元）:
                * peak_to_valley < 0.4 ms（窄波形）
                * 放电率 > 10 Hz（高频放电）
            """
            # 检查两个表格的unit_id是否完全一致
            if not df_template_metrics.index.tolist() == df_qm_metrics.index.tolist():
                raise ValueError("两表格的unit_ids不匹配")

            # 初始化分类字典：包含所有unit_id和空的细胞类型列表
            classify_dict = {"unit_ids": df_template_metrics.index.tolist(),
                             "putative_cell_type": []}

            # 判断神经元的兴奋/抑制类型
            if classify_dict["unit_ids"]:
                for unit_id in classify_dict["unit_ids"]:
                    # 将peak_to_valley从秒转换为毫秒进行比较
                    peak_to_valley_ms = df_template_metrics.loc[unit_id, 'peak_to_valley'] * 1000
                    firing_rate = df_qm_metrics.loc[unit_id, 'firing_rate']

                    # 锥体神经元：宽波形 + 低频放电
                    if (peak_to_valley_ms > 0.4) and (firing_rate < 5):
                        classify_dict["putative_cell_type"].append("putative_pyramidal_units")

                    # 中间神经元：窄波形 + 高频放电
                    elif (peak_to_valley_ms < 0.4) and (firing_rate > 10):
                        classify_dict["putative_cell_type"].append("putative_interneuron_units")
                    else:
                        # 不满足任一条件的单元，标记为未分类
                        classify_dict["putative_cell_type"].append("unclassified_units")
            return classify_dict

        sorting_analyzer = si.load(analyzer_folder)
        # 读取质量指标Excel文件
        df_qm_metrics = sorting_analyzer.extensions["quality_metrics"].get_data()
        # 执行单元筛选，获取质量标记
        renew_dict = screen_units(df_qm_metrics)
        # 获取unit所在通道的位置
        channel_dict = get_unit_max_channel(sorting_analyzer)
        renew_dict = {**renew_dict, **channel_dict}

        # 如果需要分类神经元类型
        if classify_units:
            # 读取模板指标Excel文件
            df_template_metrics = sorting_analyzer.extensions["template_metrics"].get_data()
            # 执行细胞类型分类
            classify_dict = get_units_classified(df_qm_metrics, df_template_metrics)
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

        # 保存
        yyl.check_delete_exists_path(analyzer_folder)
        sorting_analyzer.save_as(format="binary_folder", folder=analyzer_folder)

        print("Noise labels:")
        print(labels_noise)
        print("SUA/MUA labels:")
        print(labels_sua_mua)

        return labels_noise, labels_sua_mua

    def visualize_results(self, analyzer_folder, unit_id=None, fig_folder=None):
        """
        可视化结果
        Args:
            analyzer_folder: 分析器文件夹
            unit_id: 特定单元ID（如果为None则显示所有）
        """
        # 延迟导入 spikeinterface
        import spikeinterface as si
        import spikeinterface.widgets as sw
        import yyl_utils as yyl
        import matplotlib.pyplot as plt

        analyzer = si.load(analyzer_folder)

        if fig_folder:
            yyl.make_sure_folder_exist(fig_folder)

        # 打印基本信息
        unit_counts = analyzer.sorting.count_num_spikes_per_unit()
        print("unit_id:spike数")
        print({f'{x}': int(y) for (x, y) in unit_counts.items()})
        print(f"各unit通道定位：{analyzer.extensions['unit_locations'].data}")

        # 绘制波形
        sw.plot_unit_waveforms(
            sorting_analyzer_or_templates=analyzer,
            plot_channels=True,
            scalebar=True,
            backend="matplotlib"
        )
        if fig_folder:
            plt.savefig(fig_folder / "unit_waveforms.png")
            plt.close()
        # plt.show()

        # 如果有特定单元ID，绘制单元摘要
        if unit_id is not None:
            # os.environ['LANG'] = 'en_US.UTF-8'
            # os.environ['LC_ALL'] = 'en_US.UTF-8'
            sw.plot_unit_summary(
                sorting_analyzer=analyzer,
                unit_id=unit_id,
                backend="matplotlib",
                figsize=(14, 8)
            )
            if fig_folder:
                plt.savefig(fig_folder / f"unit{unit_id}_summary.png")
                plt.close()
            # plt.show()

        # sw.plot_all_amplitudes_distributions(sorting_analyser=sorting_analyser)
        # sw.plot_amplitudes(sorting_analyser=sorting_analyser)
        # 绘制每个单元的自相关图
        # sw.plot_autocorrelograms(sorting_analyser=sorting_analyser)
        # 绘制神经元对之间的互相关图
        # sw.plot_crosscorrelograms(sorting_analyser=sorting_analyser)
        # 绘制ISI分布图
        # sw.plot_isi_distribution(sorting_analyser=sorting_analyser)
        # sw.plot_quality_metrics(sorting_analyser=sorting_analyser)
        # sw.plot_rasters(sorting_analyser=sorting_analyser)
        # sw.plot_spikes_on_traces()
        # sw.plot_template_metrics()
        # sw.plot_template_similarity()
        # sw.plot_traces()
        # sw.plot_unit_presence()
        # sw.plot_unit_probe_map()
        # sw.plot_unit_templates()
        # sw.plot_unit_waveforms_density_map()
        # sw.plot_unit_waveforms()

        #     用于sorting方法对比
        # 可视化两个排序结果之间单元的匹配关系，生成一个一致性矩阵（AgreementMatrix） 或混淆矩阵（ConfusionMatrix）。
        # sw.plot_agreement_matrix

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
        import pandas as pd

        yyl.check_delete_exists_path(output_folder)

        analyzer = si.load(analyzer_folder)
        ex.export_to_phy(
            sorting_analyzer=analyzer,
            output_folder=output_folder,
            remove_if_exists=True
        )

        def label_all_good(tsv_path):
            """将Phy中的所有单元标记为good"""
            df = pd.read_csv(tsv_path, sep='\t')
            df['group'] = 'good'
            df.to_csv(tsv_path, sep='\t', index=False)

        # 修改cluster_group.tsv文件，将所有单元标记为good
        # cellexplorer只会读取标记为good的unit
        label_all_good(output_folder / 'cluster_group.tsv')

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

        pipeline = _MatlabPipeline()
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

    def run_pipeline(self, probe: Probe = None, config=None, traces_config=None, n_jobs=-1):
        """
        运行完整的spike sorting处理管道

        Args:
            config: 配置字典，包含管道运行的所有路径参数和设置
                - sorter_name (str, optional): 使用的spike sorter名称，默认值为'mountainsort5'
                - sorting_params (dict, optional): sorter-specific的排序参数，默认使用get_default_sorting_params(sorter_name)返回的参数
                - run_cell_explorer (bool, optional): 是否在管道完成后运行CellExplorer，默认值为False

            probe: 定义探针，可以用self.get_probe获得

            traces_config: 可选字典，当需要从numpy数组而不是原始文件创建recording时提供
                需严格按照以下键值对形成字典，其中traces、fs、chan_ids是必须的，
                示例：{'traces':all_WBCs,'fs':fs,'chan_ids':channels,}
                - traces (np.ndarray): 神经信号数据数组，形状为(n_samples, n_channels)
                - fs (float): 采样频率(Hz)
                - chan_ids (list): 通道ID列表
                - metadata_folder (str): 包含recording元数据的文件夹路径(可选)
                - properties (dict): recording属性字典(可选)

            n_jobs: 并行数

        Workflow:
            1. 保存recording文件 (从PLX文件或numpy数组)
            2. 预处理数据 (滤波、去噪等)
            3. 执行spike sorting
            4. 创建分析器并计算质量指标
            5. 可视化排序结果
            6. 导出到Phy格式
            7. (可选) 运行CellExplorer进行进一步分析

        Note:
            - 当traces_config为None时，从self.input_file读取PLX文件
            - 当traces_config提供时，从numpy数组创建recording
            - MATLAB_AVAILABLE需要为True才能运行CellExplorer
        """

        if config is None:
            config = {}
        if traces_config is None:
            traces_config = {}
        # 解包配置
        sorter_name = config.get('sorter_name', 'mountainsort5')
        sorting_params = config.get('sorting_params', self.get_default_sorting_params(sorter_name))
        run_cell_explorer = config.get('run_cell_explorer', False)

        # 执行管道步骤
        print("步骤 1/6: 保存recording文件...")
        if traces_config:
            # 从numpy数组创建并保存recording
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
            # 从PLX文件读取并保存recording
            self.read_save_plx_file(
                self.input_file,
                self.output_paths.raw_recording_folder,
                probe=probe,
                n_jobs=n_jobs,
            )

        print("步骤 2/6: 预处理数据...")
        # 执行数据预处理(滤波、whitening等)
        self.preprocess_recording(
            self.output_paths.raw_recording_folder,
            self.output_paths.preprocessed_recording_folder,
            n_jobs=n_jobs,
        )

        print("步骤 3/6: 执行排序...")
        # 运行spike sorting算法
        self.perform_sorting(
            self.output_paths.preprocessed_recording_folder,
            sorter_name,
            self.output_paths.sorting_object_folder,
            sorting_params
        )

        print("步骤 4/6: 创建分析器...")
        # 创建SortingAnalyzer并计算质量指标
        self.create_analyzer(
            self.output_paths.preprocessed_recording_folder,
            self.output_paths.sorting_object_folder,
            self.output_paths.sorting_analyzer_folder,
            self.output_paths.qm_excel_path,
            n_jobs=n_jobs,
        )
        # 获取unit的类型、sorting质量如何和最大通道等指标
        self.renew_unit_type(
            self.output_paths.sorting_analyzer_folder,
            self.output_paths.cell_type_metrics_path,
        )

        print("步骤 5/6: 可视化结果...")
        # 生成排序结果的可视化图表
        self.visualize_results(
            self.output_paths.sorting_analyzer_folder,
            fig_folder=self.output_paths.figures_folder
        )

        print("步骤 6/6: 导出到Phy...")
        # 导出为Phy格式用于手动curation
        self.export_to_phy(
            self.output_paths.sorting_analyzer_folder,
            self.output_paths.phy_folder
        )

        # 可选：运行CellExplorer进行进一步分析
        if run_cell_explorer and self._matlab_available:
            print("运行CellExplorer...")
            matlab_func_path = self.matlab_func_path
            metrics_path = self.output_paths.cell_metrics_path
            self.run_cell_explorer(
                self.output_paths.phy_folder,
                matlab_func_path,
                metrics_path
            )

        print("管道执行完成!")

def print_sorter_params(sorter_name):
    """
    获取某个sorting方法的详细参数和描述
    :param sorter_name:sorting方法名称
    """
    from spikeinterface.sorters import get_sorter_params_description,get_default_sorter_params
    print(get_default_sorter_params(sorter_name))
    description = get_sorter_params_description(sorter_name)
    for param_name, param_desc in description.items():
        print(f"{param_name}: {param_desc}")

def print_spikeinterface_sorters():
    """
    打印spikeinterface支持的所有sorting方法
    """
    from spikeinterface.sorters import available_sorters
    print(available_sorters())

def print_available_sorters():
    """
    打印目前环境下可用的所有sorter名称
    """
    import os
    # 设置CMD为UTF-8模式
    os.system('chcp 65001')  # 设置控制台为UTF-8
    from spikeinterface.sorters import installed_sorters
    print(installed_sorters())