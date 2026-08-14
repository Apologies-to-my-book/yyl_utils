from pathlib import Path
from typing import Union, List
import shutil
import os
import numpy as np



class IntervalOps:
    """
    整数区间操作工具类

    所有区间均定义为左闭右闭 [start, end]，且区间端点均为整数。
    提供区间的交、并、差、合并、筛选等操作，支持 np.ndarray 和 list 两种输入格式。
    注意：
    1.这个代码的操作都是对区间进行的，意味着例如求差集，会自动把相邻的区间合并，并且返回差集，而不是返回完全不互包的元素，所以你如果要求不互包的元素，
    不要用这个代码，这个代码会返回各个小区间被切以后的结果。
    2.注意这个代码的操作对象均为整数区间，也就是所有点只考虑整数，且是左闭右闭的。
    """

    @staticmethod
    def _ensure_list(intervals):
        """
        将输入统一转换为 list 格式，兼容 np.ndarray

        参数:
            intervals: list 或 np.ndarray，如 [[1, 3], [5, 7]] 或 np.array([[1, 3], [5, 7]])
        返回:
            list 格式的区间列表
        """
        if isinstance(intervals, np.ndarray):
            return intervals.tolist()
        return intervals

    @classmethod
    def intersect(cls, list_of_intervals):
        """
        求多个区间列表的交集（所有列表共同覆盖的区域）

        使用扫描线算法：记录每个区间的进入和离开事件，当某个位置同时被所有列表
        覆盖时（coverage == n_lists），该位置属于交集。

        参数:
            list_of_intervals: 包含多个区间列表的列表，如 [
                [[1, 5], [8, 10]],
                [[3, 7], [9, 12]],
                [[2, 4], [6, 9]]
            ]
        返回:
            所有列表共同覆盖的区间列表，如 [[3, 4], [9, 9]]
            区间为左闭右闭 [start, end]

        时间复杂度: O(N log N)，N 为所有区间的总端点数量
        """
        if not list_of_intervals:
            return []

        n_lists = len(list_of_intervals)
        events = []

        # 为每个区间的起点和终点生成事件
        # 起点事件: (start, 1)，表示进入区间
        # 终点事件: (end+1, -1)，表示离开区间（因为整数区间右闭，所以在 end+1 处离开）
        for intervals in list_of_intervals:
            intervals = cls._ensure_list(intervals)
            for start, end in intervals:
                if start <= end:
                    events.append((start, 1))
                    events.append((end + 1, -1))

        # 按位置排序，位置相同时先处理进入事件（1 > -1），确保在边界处正确计数
        events.sort(key=lambda x: (x[0], x[1]))

        result = []
        coverage = 0  # 当前位置被多少个列表覆盖
        start = None  # 交集区间的起始位置

        for pos, delta in events:
            # 如果之前处于完全覆盖状态（coverage == n_lists），且当前位置大于起始位置，
            # 说明 [start, pos-1] 这一段属于交集
            if coverage == n_lists and start is not None and pos > start:
                result.append([start, pos - 1])

            # 更新覆盖计数
            coverage += delta

            # 如果此时达到完全覆盖状态，记录新的区间起始位置
            # 否则清空起始位置，表示当前不在交集中
            if coverage == n_lists:
                start = pos
            else:
                start = None

        return result

    @classmethod
    def subtract(cls, intervals_a, intervals_b):
        """
        求区间差集 A - B：返回在 A 中但不在 B 中的区间

        使用扫描线算法：A 的区间贡献正覆盖，B 的区间贡献负覆盖，
        当某个位置的净覆盖 > 0 时，该位置属于差集。

        参数:
            intervals_a: 被减区间列表，如 [[1, 10], [15, 20]]
            intervals_b: 减去的区间列表，如 [[3, 5], [8, 12]]
        返回:
            差集区间列表，如 [[1, 2], [6, 7], [15, 20]]
            区间为左闭右闭 [start, end]

        时间复杂度: O(N log N)，N 为两个列表的区间总端点数量
        """
        intervals_a = cls._ensure_list(intervals_a)
        intervals_b = cls._ensure_list(intervals_b)

        # 如果 B 为空，直接返回 A
        if not intervals_b:
            return [list(interval) for interval in intervals_a]

        # 先合并 A 中的重叠区间，简化后续处理
        merged_a = cls.merge_overlap(intervals_a)

        events = []

        # A 的区间贡献正覆盖（delta=1）
        for start, end in merged_a:
            if start <= end:
                events.append((start, 1))
                events.append((end + 1, -1))

        # B 的区间贡献负覆盖（delta=-1）
        for start, end in intervals_b:
            if start <= end:
                events.append((start, -1))
                events.append((end + 1, 1))

        # 按位置排序，位置相同时先处理正覆盖事件
        events.sort(key=lambda x: (x[0], x[1]))

        result = []
        coverage = 0  # 当前位置的净覆盖计数
        start = None  # 差集区间的起始位置

        for pos, delta in events:
            # 如果之前处于正覆盖状态（coverage > 0），且当前位置大于起始位置，
            # 说明 [start, pos-1] 这一段属于差集
            if coverage > 0 and start is not None and pos > start:
                result.append([start, pos - 1])

            # 更新覆盖计数
            coverage += delta
            # 如果净覆盖 > 0，记录新区间的起始位置
            start = pos if coverage > 0 else None

        return result

    @classmethod
    def union(cls, list_of_intervals):
        """
        求多个区间列表的并集，合并所有重叠或相邻的区间

        将所有区间展平后排序，然后扫描合并。相邻区间（如 [1,5] 和 [6,10]）
        会被合并为 [1,10]，因为端点都是整数，这两个区间实际上是连续的。

        参数:
            list_of_intervals: 包含多个区间列表的列表，如 [
                [[1, 3], [5, 7]],
                [[2, 4], [6, 8]]
            ]
        返回:
            合并后的并集区间列表，如 [[1, 4], [5, 8]]
            区间为左闭右闭 [start, end]

        时间复杂度: O(N log N)，N 为所有区间的总数量
        """
        # 将所有区间收集到一个列表中
        all_intervals = []
        for intervals in list_of_intervals:
            intervals = cls._ensure_list(intervals)
            for interval in intervals:
                all_intervals.append([interval[0], interval[1]])

        if len(all_intervals) == 0:
            return []

        # 按起点排序
        all_intervals.sort(key=lambda x: x[0])

        merged = [all_intervals[0][:]]  # 用切片创建副本，避免修改原数据

        # 遍历剩余区间，与合并结果中的最后一个区间比较
        for current in all_intervals[1:]:
            last = merged[-1]
            curr_start, curr_end = current[0], current[1]

            # 如果当前区间的起点 <= 最后一个区间的终点 + 1，
            # 说明两个区间重叠或相邻（整数情况下相邻即连续），需要合并
            if curr_start <= last[1] + 1:
                merged[-1][1] = max(last[1], curr_end)
            else:
                merged.append([curr_start, curr_end])

        return merged

    @classmethod
    def merge_overlap(cls, intervals):
        """
        合并重叠或相邻的区间

        与 union 类似，但输入是单个区间列表而不是多个列表的列表。
        相邻区间（如 [1,5] 和 [6,10]）会被合并为 [1,10]。

        参数:
            intervals: 区间列表，如 [[1, 3], [2, 4], [6, 8]]
        返回:
            合并后的区间列表，如 [[1, 4], [6, 8]]
            区间为左闭右闭 [start, end]

        时间复杂度: O(N log N)，N 为区间数量
        """
        intervals = cls._ensure_list(intervals)

        if not intervals:
            return []

        # 按起点排序
        sorted_intervals = sorted(intervals)
        merged = []
        current_start, current_end = sorted_intervals[0]

        # 遍历剩余区间，逐个合并
        for start, end in sorted_intervals[1:]:
            # 如果当前区间与正在合并的区间重叠或相邻
            if start <= current_end + 1:
                current_end = max(current_end, end)
            else:
                # 否则保存当前合并结果，开始新的合并
                merged.append([current_start, current_end])
                current_start, current_end = start, end

        # 保存最后一个合并结果
        merged.append([current_start, current_end])
        return merged

    @classmethod
    def filter_points_in_interval_numpy(cls, points, interval_b):
        """
        筛选时间点列表 A 中所有落在区间 B 内的点（NumPy 向量化版本）

        使用 NumPy 的向量化运算一次性对所有点进行区间判断，
        避免 Python 层面的显式循环，在处理大量点时性能优异。

        参数:
            points: 时间点列表 A，可以是 list 或 np.ndarray
                    如 [1, 3, 5, 7, 9, 11, 13]
            interval_b: 区间列表 B，如 [[2, 4], [6, 8], [10, 12]]
        返回:
            A 中所有落在 B 区间内的点（list 格式）
            如 [3, 7, 11]

        实现原理:
            1. 先合并 B 区间，减少判断次数
            2. 将 points 转换为 NumPy 数组
            3. 创建布尔掩码（全 False），对每个合并后的 B 区间，
               使用向量化比较更新掩码，标记落在该区间内的点
            4. 使用掩码提取满足条件的点

        时间复杂度: O(M * K)，M 为合并后 B 的区间数，K 为点的数量
                   由于 NumPy 向量化操作在底层用 C 实现，实际运行速度远快于
                   纯 Python 循环的 O(N * M)
        空间复杂度: O(K)，用于存储掩码和结果
        """
        interval_b = cls._ensure_list(interval_b)

        # 边界检查：点列表或区间列表为空时直接返回空列表
        if len(points) == 0 or not interval_b:
            return []

        # 先合并 B 区间，减少需要判断的区间数量
        merged_b = cls.merge_overlap(interval_b)

        # 将点列表转换为 NumPy 数组，便于向量化操作
        points = np.asarray(points)

        # 创建布尔掩码，初始全为 False（所有点都不在区间内）
        mask = np.zeros(len(points), dtype=bool)

        # 对每个合并后的 B 区间，使用向量化比较更新掩码
        # points >= b_start 返回布尔数组，表示每个点是否 >= 区间起点
        # points <= b_end 返回布尔数组，表示每个点是否 <= 区间终点
        # & 运算得到同时满足两个条件的点（即在区间内）
        # |= 运算将当前区间的结果合并到总掩码中
        for b_start, b_end in merged_b:
            mask |= (points >= b_start) & (points <= b_end)

        # 使用掩码提取满足条件的点，并转换为 list 返回
        return points[mask].tolist()

class SettingsWithSave():
    @staticmethod
    def save_settings_generic(settings_obj, output_path: Path, extra_info: dict = None):
        """
        通用保存配置（支持任意嵌套对象）
        自动根据文件扩展名选择保存格式：.json 或 .log
        """
        import json
        from datetime import datetime

        def to_serializable(obj):
            """将任意对象转换为可JSON序列化的格式"""
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in vars(obj).items():
                    if not key.startswith("_"):
                        result[key] = to_serializable(value)
                return result
            elif isinstance(obj, (list, tuple)):
                return [to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (Path, datetime)):
                return str(obj)
            else:
                return obj

        # 转换为可序列化的字典
        config_dict = to_serializable(settings_obj)

        if extra_info:
            config_dict["extra_info"] = extra_info

        config_dict["saved_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 根据文件扩展名选择保存格式
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()

        if suffix == '.json':
            # 保存为JSON格式
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)

        elif suffix == '.log':
            # 保存为LOG格式（更易读的文本日志格式）
            SettingsWithSave._save_as_log(config_dict, output_path)

        else:
            # 默认保存为JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
            print(f"⚠️ 未知文件扩展名 '{suffix}'，默认保存为JSON格式")

    @staticmethod
    def _save_as_log(config_dict, output_path: Path):
        """
        将配置保存为.log格式（易读的文本日志格式）
        """
        from datetime import datetime

        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入标题头
            f.write("=" * 80 + "\n")
            f.write(f"  CONFIGURATION SNAPSHOT\n")
            f.write(f"  Saved: {config_dict.get('saved_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n")
            f.write("=" * 80 + "\n\n")

            # 递归写入配置内容
            SettingsWithSave._write_dict_to_log(f, config_dict, indent_level=0)

            f.write("\n" + "=" * 80 + "\n")
            f.write("  END OF CONFIGURATION\n")
            f.write("=" * 80 + "\n")

    @staticmethod
    def _write_dict_to_log(file_handler, data_dict, indent_level=0):
        """
        递归将字典写入log文件（支持嵌套结构）
        """
        indent = "  " * indent_level

        for key, value in data_dict.items():
            # 跳过元数据（已在头部显示）
            if key in ["saved_time"]:
                continue

            if isinstance(value, dict):
                # 嵌套字典
                file_handler.write(f"{indent}[{key}]\n")
                SettingsWithSave._write_dict_to_log(file_handler, value, indent_level + 1)
            elif isinstance(value, list):
                # 列表
                file_handler.write(f"{indent}{key}: [\n")
                for item in value:
                    if isinstance(item, dict):
                        SettingsWithSave._write_dict_to_log(file_handler, item, indent_level + 1)
                    else:
                        file_handler.write(f"{indent}  {item},\n")
                file_handler.write(f"{indent}]\n")
            else:
                # 普通键值对
                file_handler.write(f"{indent}{key}: {value}\n")

def add_suffix_to_filename(filepath: Union[str, Path], suffix: str) -> Union[str, Path]:
    """
    在文件名后添加后缀，保持输入类型

    Args:
        filepath: 原始文件路径，可以是字符串或Path对象
        suffix: 要添加的后缀

    Returns:
        添加后缀后的新文件路径（返回类型与输入类型一致）

    Examples:
        # >>> add_suffix_to_filename("/path/to/file.txt", "backup")
        '/path/to/file_backup.txt'
        # >>> add_suffix_to_filename(Path("/path/to/file.txt"), "backup")
        Path('/path/to/file_backup.txt')
    """
    # 统一转换为Path对象进行处理
    path_obj = Path(filepath) if isinstance(filepath, str) else filepath

    # 分离文件名和扩展名
    name = path_obj.stem  # 文件名（不含扩展名）
    ext = path_obj.suffix  # 扩展名（包含点）

    # 构建新文件名：原文件名_后缀.扩展名
    new_filename = f"{name}_{suffix}{ext}"

    # 组合成完整路径
    new_filepath = path_obj.parent / new_filename

    # 根据输入类型返回相应类型
    return str(new_filepath) if isinstance(filepath, str) else new_filepath


def make_sure_folder_exist(folder: Union[str, Path, list[Union[str, Path]]]) -> Union[Path, list[Path]]:
    """
    确保文件夹存在，如果不存在则创建。
    参数:
        folder: 可以是字符串、Path对象或它们的列表
    返回:
        创建的文件夹路径（单个Path或Path列表）
    """
    def create_single_folder(folder_path):
        """创建单个文件夹并返回Path对象"""
        path = Path(folder_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path
    # 处理输入参数
    if isinstance(folder, (str, Path)):
        # 单个路径
        result = create_single_folder(folder)
    elif isinstance(folder, list):
        # 路径列表
        result = []
        for item in folder:
            result.append(create_single_folder(item))
    else:
        raise TypeError(f"不支持的类型: {type(folder)}")

    return result


def isolate(func):
    """
    闭包函数装饰器，功能是让函数只能使用局部变量，不能使用全局变量
    示例：
    @isolate
    def my_function():
        print(x)  # 这里会报错，因为 x 不存在

    x = 10
    my_function()
    """
    def wrapper(*args, **kwargs):
        import builtins
        # 备份当前全局变量
        original_globals = globals().copy()
        try:
            # 清空全局变量
            globals().clear()
            globals().update({'__builtins__': builtins})
            # 调用函数
            return func(*args, **kwargs)
        finally:
            # 恢复全局变量
            globals().clear()
            globals().update(original_globals)
    return wrapper

def check_delete_exists_path(path: Union[str, Path, List[Union[str, Path]]]):
    '''
    检查路径（文件或文件夹）是否存在，若存在，则删除它
    :param path: 单个路径或路径列表，支持字符串和Path对象
    :return: path 返回path对象
    '''
    paths = [path] if isinstance(path, (str, Path)) else path

    for p in paths:
        p_path = Path(p) if isinstance(p, str) else p
        if p_path.exists():
            if p_path.is_file():
                os.remove(p_path)  # 删除文件
            elif p_path.is_dir():
                shutil.rmtree(p_path)  # 删除文件夹
    return path


def matlab_struct_to_dict(matlab_obj):
    """
    将matlab中的struct对象转换成python的dict，使用的时候先用scipy.io加载该对象，再用这个函数
        data = sio.loadmat('data.mat', squeeze_me=True, struct_as_record=False)
        matlab_struct = data['structVar']  # matlab中的变量名
        python_dict = matlab_struct_to_dict(matlab_struct)
    :param matlab_obj:
    :return:返回一个python的dict
    """
    # 如果是 MATLAB struct 对象
    if hasattr(matlab_obj, '_fieldnames'):
        return {field: matlab_struct_to_dict(getattr(matlab_obj, field))
                for field in matlab_obj._fieldnames}

    # 如果是 numpy 数组
    elif isinstance(matlab_obj, np.ndarray):
        # 如果是 object 数组（可能包含 struct）
        if matlab_obj.dtype == np.dtype('object'):
            # 递归处理数组中的每个元素
            if matlab_obj.size == 1:
                return matlab_struct_to_dict(matlab_obj.item())
            else:
                return [matlab_struct_to_dict(item) for item in matlab_obj]

        # 如果是普通数值数组
        else:
            return matlab_obj.tolist() if matlab_obj.size == 1 else matlab_obj.tolist()

    # 其他类型直接返回
    else:
        return matlab_obj


def delete_path(path):
    """删除文件或目录（包括非空目录）"""
    path = Path(path)

    if not path.exists():
        print(f"路径不存在: {path}")
        return False

    if path.is_file():
        path.unlink()
        print(f"已删除文件: {path}")
    elif path.is_dir():
        shutil.rmtree(path)
        print(f"已删除目录: {path}")

    return True