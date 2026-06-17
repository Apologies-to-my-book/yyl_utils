from pathlib import Path
from typing import Union, List
import shutil
import os
import numpy as np

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