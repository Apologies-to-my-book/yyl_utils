import warnings
import cProfile
import pstats
import inspect
import os
import functools
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Callable
import time


def print_dir(dir_path: Union[str, Path], print_subdir: bool = True, pattern: Union[str, None] = None):
    """
    打印目录内容，支持通配符过滤

    Args:
        dir_path: 目标目录路径
        print_subdir: True=遍历所有子目录，False=仅当前目录
        pattern: 通配符模式（如 "*.txt", "*.py"），None=所有内容
    """
    from pathlib import Path
    dir_path = Path(dir_path)

    if print_subdir:
        # 递归遍历所有子目录
        if pattern:
            # 匹配指定模式的文件（只返回文件，排除目录本身）
            for item in dir_path.rglob(pattern):
                if item.is_file():
                    print(str(item))
        else:
            # 递归获取所有文件（排除目录）
            for item in dir_path.rglob("*"):
                if item.is_file():
                    print(str(item))
    else:
        # 仅遍历当前目录
        if pattern:
            for item in dir_path.glob(pattern):
                if item.is_file():
                    print(str(item))
        else:
            for item in dir_path.glob("*"):
                if item.is_file():
                    print(str(item))


def get_full_size(obj):
    """
    使用 pympler 库的 asizeof 函数，递归计算对象及其所有引用对象的总内存占用。
    这是最简单、最准确的完整内存大小检测方法。
    参数:
        obj: any - 任意Python对象（列表、字典、自定义对象等）
    返回:
        int - 对象的完整内存占用大小（单位：字节）
    示例:
        # >>> my_list = [1, 2, 3, "hello", [4, 5, 6]]
        # >>> size = get_full_size(my_list)
        # >>> print(f"完整内存占用: {size} bytes")
        完整内存占用: 456 bytes
    """
    from pympler import asizeof
    return asizeof.asizeof(obj)

def analyze_obj_memory(sort_by='size', limit=20, verbose=False, scope="all"):
    """
    详细分析内存使用情况，返回DataFrame

    Args:
        sort_by: 排序方式 ('size', 'name')
        limit: 显示的数量限制
        verbose: 是否打印详细报告
        scope: 分析范围 ('all', 'local', 'global')，分别是全部、仅局部、仅全局变量
    """
    import sys
    import inspect
    from pympler import asizeof
    import pandas as pd

    # 正确获取调用者的帧
    # 方法1：使用 inspect（推荐）
    frame = inspect.currentframe().f_back

    # 方法2：使用 sys._getframe（也可以）
    # frame = sys._getframe(1)

    # 根据scope参数选择要分析的变量
    if scope == "local":
        vars_dict = frame.f_locals
    elif scope == "global":
        vars_dict = frame.f_globals
    else:  # all
        vars_dict = {}
        vars_dict.update(frame.f_globals)
        vars_dict.update(frame.f_locals)

    memory_data = []
    for var_name, var_value in vars_dict.items():
        if (isinstance(var_name, str) and
                not var_name.startswith('__') and
                var_name not in ['__file__', '__name__']):
            try:
                size_bytes = asizeof.asizeof(var_value)
                memory_data.append({
                    'name': var_name,
                    'size_bytes': size_bytes,
                    'size_kb': size_bytes / 1024,
                    'size_mb': size_bytes / 1024 / 1024,
                    'type': type(var_value).__name__
                })
            except:
                pass

    # 清理帧引用以避免循环引用
    del frame

    df = pd.DataFrame(memory_data)
    if df.empty:
        if verbose:
            print("未找到可分析的变量")
        return df

    # 排序和限制
    if sort_by == 'size':
        df = df.sort_values('size_bytes', ascending=False)
    elif sort_by == 'name':
        df = df.sort_values('name')
    df = df.head(limit)

    # 只有在verbose为True时才打印
    if verbose:
        print("内存使用分析报告:")
        print(f"分析范围: {scope}")
        print("=" * 80)
        for _, row in df.iterrows():
            print(f"{row['name']:<25} {row['size_mb']:>8.2f} MB {row['type']:<15}")
        total_mb = df['size_mb'].sum()
        print("=" * 80)
        print(f"前 {len(df)} 个变量总内存: {total_mb:.2f} MB")

    return df

def redirect_print_to_log(log_file: str, mode: str = 'a'):
    """
    装饰器函数，将函数内的所有print重定向到log文件
    Args:
        log_file: 日志文件路径
        mode: 文件打开模式，'a'为追加，'w'为覆盖
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 重定向stdout到文件
            original_stdout = sys.stdout
            try:
                with open(log_file, mode, encoding='utf-8') as f:
                    sys.stdout = f  # 重定向stdout到文件
                    result = func(*args, **kwargs)
                    return result
            finally:
                sys.stdout = original_stdout  # 恢复stdout
        return wrapper
    return decorator

def ipdb_debug(enable_warnings=False, warning_categories=None, enable_exceptions=True, skip_exceptions=None):
    """
    设置全局异常钩子，在报错时自动进入 ipdb 调试模式
    使用的时候只需要导入该函数并在代码中调用该函数即可，即ipdb_debug()
    如果需要在某行设置断点，则import ipdb后，在该行输入ipdb.set_trace()
    常用debug命令：
    最常用命令（记住这些就够用了）
        n  - 下一步（不进入函数）
        s  - 进入函数内部
        c  - 继续执行程序
        p  - 打印变量值
        l  - 查看当前代码
        w  - 查看调用栈
        q  - 退出调试模式
    基础导航命令
        n / next           - 执行下一行（不进入函数）
        s / step           - 进入函数调用
        c / continue       - 继续执行直到下一个断点
        r / return         - 继续执行直到当前函数返回
        unt [行号]         - 执行直到指定行号
        j [行号]           - 跳转到指定行号
    代码查看命令
        l / list           - 查看当前行周围的代码
        ll / longlist      - 查看当前函数的完整源代码
        w / where          - 打印调用栈跟踪
        u / up             - 在调用栈中向上移动
        d / down           - 在调用栈中向下移动
        source [对象]      - 显示对象的源代码
    变量查看命令
        p [表达式]         - 打印表达式的值
        pp [表达式]        - 漂亮打印（格式化输出）
        whatis [表达式]    - 打印表达式的类型
        display [表达式]   - 每次停止时自动显示
        ptype [表达式]     - 打印详细类型信息
    断点管理命令
        b [行号]           - 在指定行设置断点
        b [函数名]         - 在函数入口设置断点
        b [条件]           - 设置条件断点
        cl                 - 清除所有断点
        cl [编号]          - 清除指定断点
        disable [编号]     - 禁用断点
        enable [编号]      - 启用断点
    信息查询命令
        h / help           - 显示帮助信息
        help [命令]        - 显示具体命令帮助
        pinfo [变量]       - 显示变量详细信息
        %who               - 显示所有变量
        %whos              - 显示变量详细信息
    执行控制命令
        c / continue       - 继续执行程序
        q / quit           - 退出调试并终止程序
        Ctrl + D           - 退出调试（快捷键）
    实用技巧命令
        pp locals()        - 漂亮打印所有局部变量
        pp globals()       - 漂亮打印所有全局变量
        !变量=值           - 临时修改变量值
        %timeit [代码]     - 测量代码执行时间
    Args:
        enable_warnings: 是否在警告时进入调试模式
        warning_categories: 要捕获的警告类型列表，默认为所有警告
        enable_exceptions: 是否在异常时进入调试模式
        skip_exceptions: 要跳过的异常类型列表
    """
    import ipdb
    if skip_exceptions is None:
        skip_exceptions = [KeyboardInterrupt, SystemExit]
    if warning_categories is None:
        warning_categories = [Warning]  # 捕获所有警告

    original_showwarning = warnings.showwarning
    # 添加一个集合来记录要忽略的警告位置
    ignored_warning_locations = set()

    def custom_showwarning(message, category, filename, lineno, file=None, line=None):
        """
        自定义警告显示函数，在警告时进入调试模式
        """
        # 先调用原始警告显示，确保警告信息正常输出
        original_showwarning(message, category, filename, lineno, file, line)

        # 检查这个警告位置是否已经被忽略
        warning_location = f"{filename}:{lineno}"
        if warning_location in ignored_warning_locations:
            return  # 直接返回，不进入调试

        if enable_warnings and any(issubclass(category, warn_type) for warn_type in warning_categories):
            print("\n" + "=" * 60)
            print("触发警告：")
            print(f"警告类型: {category.__name__}")
            print(f"警告信息: {message}")
            print(f"文件: {filename}:{lineno}")
            print("=" * 60)

            # 交互式选择
            while True:
                user_input = input(
                    "是否进入调试模式？ (y-进入调试 / n-忽略此位置的所有警告 / c-继续执行): ").strip().lower()
                if user_input == 'y':
                    # 进入调试模式
                    print("进入 ipdb 调试模式...")
                    import inspect
                    frame = inspect.currentframe()
                    # 向上追溯找到触发警告的帧
                    for i in range(5):  # 最多追溯5层
                        frame = frame.f_back
                        if frame is None:
                            break
                        # 检查这个帧是否在触发警告的文件中
                        if frame.f_code.co_filename == filename and frame.f_lineno == lineno:
                            break
                    if frame:
                        ipdb.set_trace(frame)
                    else:
                        ipdb.set_trace()
                    break
                elif user_input == 'n':
                    # 忽略此位置的所有未来警告
                    ignored_warning_locations.add(warning_location)
                    print(f"已忽略此位置的警告: {warning_location}")
                    print("后续相同位置的警告将自动跳过")
                    break
                elif user_input == 'c':
                    # 继续执行，不进入调试
                    print("继续执行程序...")
                    break
                else:
                    print("请输入 y, n 或 c")
    def ipdb_on_exception(exctype, value, tb):
        """
        异常处理函数
        """
        if enable_exceptions and not any(issubclass(exctype, skip_type) for skip_type in skip_exceptions):
            print("\n" + "=" * 60)
            print("发生异常，进入调试模式：")
            # 打印异常信息
            traceback.print_exception(exctype, value, tb)
            print("=" * 60)
            # 进入 ipdb 调试模式
            ipdb.post_mortem(tb)
        else:
            # 使用默认异常处理
            sys.__excepthook__(exctype, value, tb)
    # 设置全局钩子
    if enable_warnings:
        warnings.showwarning = custom_showwarning
        # 捕获所有指定类型的警告
        for warning_category in warning_categories:
            warnings.simplefilter('always', warning_category)
    if enable_exceptions:
        sys.excepthook = ipdb_on_exception
    print("ipdb 调试模式已启用")
    if enable_warnings:
        print(f"   - 警告调试: 已启用 (捕获: {[w.__name__ for w in warning_categories]})")
        print("   - 交互选项: y-进入调试 / n-忽略此位置 / c-继续执行")
    if enable_exceptions:
        print(f"   - 异常调试: 已启用 (跳过: {[e.__name__ for e in skip_exceptions]})")

def ipdb_trace():
    import ipdb
    return ipdb.set_trace()

def decorator_func_error(
        log_file: Optional[Union[str, Callable]] = None,
        print_error: bool = True,
        raise_exception: bool = False,
        include_traceback: bool = True,
        log_success: bool = False,
        log_all_output: bool = False,
        max_retries: int = 0,  # 新增：最大重试次数，0表示不重试
        retry_delay: float = 1.0,  # 新增：重试延迟时间（秒）
        ignore_exceptions: tuple = (),  # 新增：忽略的异常类型（遇到这些异常不重试，直接失败）
        on_retry: Optional[Callable] = None,  # 新增：重试前的回调函数
        continue_on_error: bool = True  # 新增：出错后是否继续执行（不停止程序）
):
    """
    增强版错误处理装饰器，支持错误重试和完整的日志记录功能

    Args:
        log_file: 日志文件路径，可以是字符串或返回字符串的函数
        print_error: 是否在控制台打印错误信息
        raise_exception: 是否重新抛出异常
        include_traceback: 是否包含详细的堆栈跟踪信息
        log_success: 是否在函数成功执行时也记录日志
        log_all_output: 是否重定向所有输出到日志文件（不影响控制台输出）
        max_retries: 最大重试次数（0表示不重试）
        retry_delay: 重试延迟时间（秒）
        ignore_exceptions: 忽略的异常类型（遇到这些异常不重试，直接失败）
        on_retry: 重试前的回调函数，接受参数 (attempt, exception, delay)
        continue_on_error: 出错后是否继续执行（返回None而不是停止程序）

    Example:
        # 基础使用
        @decorator_func_error(max_retries=3, retry_delay=2.0)
        def unstable_function():
            # 可能失败的操作
            pass

        # 忽略特定异常
        @decorator_func_error(
            max_retries=3,
            ignore_exceptions=(ValueError,),
            continue_on_error=True
        )
        def process_data(data):
            # 处理数据，ValueError不重试直接返回None
            pass
    """

    def _write_log(log_file_path: str, message: str, level: str = "INFO"):
        """内部辅助函数：写入日志文件"""
        try:
            log_dir = Path(log_file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            with open(log_file_path, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] [{level}] {message}\n")
        except Exception as log_error:
            print(f"写入日志文件失败: {log_error}")

    class TeeOutput:
        """同时输出到控制台和日志文件的类"""

        def __init__(self, original, log_file_path):
            self.original = original
            self.log_file_path = log_file_path

        def write(self, text):
            # 输出到控制台
            self.original.write(text)
            self.original.flush()
            # 同时写入日志文件
            if text.strip():  # 只写入非空内容
                try:
                    with open(self.log_file_path, 'a', encoding='utf-8') as f:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{timestamp}] [OUTPUT] {text}")
                except Exception:
                    pass  # 忽略日志写入错误

        def flush(self):
            self.original.flush()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # === 动态日志路径处理 ===
            actual_log_file = log_file
            if callable(log_file):
                actual_log_file = log_file(*args, **kwargs)

            # === 输出重定向设置 ===
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            # === 开始执行函数 ===
            start_time = datetime.now()
            function_name = func.__name__

            # 记录函数开始执行
            if actual_log_file and log_all_output:
                _write_log(actual_log_file, f"函数 {function_name} 开始执行", "INFO")

            # 重试逻辑
            last_exception = None
            attempt = 0

            while attempt <= max_retries:
                try:
                    attempt += 1

                    # 如果不是第一次尝试，输出重试信息
                    if attempt > 1:
                        retry_info = f"第 {attempt - 1} 次重试 {function_name}..."
                        if print_error:
                            print(retry_info)
                        if actual_log_file:
                            _write_log(actual_log_file, retry_info, "RETRY")

                    # 如果需要重定向所有输出
                    if actual_log_file and log_all_output:
                        sys.stdout = TeeOutput(original_stdout, actual_log_file)
                        sys.stderr = TeeOutput(original_stderr, actual_log_file)

                    # 执行被装饰的原始函数
                    result = func(*args, **kwargs)

                    # 恢复标准输出
                    if actual_log_file and log_all_output:
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr

                    # 计算执行时间
                    end_time = datetime.now()
                    execution_time = (end_time - start_time).total_seconds()

                    # 如果重试后成功，记录成功信息
                    if attempt > 1:
                        success_msg = (f"函数 {function_name} 在第 {attempt} 次尝试时执行成功 - "
                                       f"总耗时: {execution_time:.2f}秒")
                    else:
                        success_msg = f"函数 {function_name} 执行成功 - 耗时: {execution_time:.2f}秒"

                    # 记录成功日志
                    if actual_log_file and log_success:
                        _write_log(actual_log_file, success_msg, "SUCCESS")

                    if print_error and attempt > 1:
                        print(success_msg)

                    return result

                except Exception as e:
                    # 恢复标准输出
                    if actual_log_file and log_all_output:
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr

                    last_exception = e

                    # 检查是否为需要忽略的异常
                    if isinstance(e, ignore_exceptions):
                        if print_error:
                            print(f"遇到忽略的异常 {type(e).__name__}，停止重试")
                        break

                    # 计算当前尝试的执行时间
                    current_end_time = datetime.now()
                    current_execution_time = (current_end_time - start_time).total_seconds()

                    # 构造错误信息
                    error_msg = f"函数 {function_name} 第 {attempt} 次尝试失败 - 已耗时: {current_execution_time:.2f}秒"
                    detailed_error_msg = f"{error_msg}\n错误详情: {str(e)}"

                    # 包含堆栈跟踪
                    if include_traceback:
                        tb_str = traceback.format_exc()
                        detailed_error_msg = f"{detailed_error_msg}\n\n完整堆栈跟踪:\n{tb_str}"

                    # 控制台输出错误信息
                    if print_error:
                        print(f"{detailed_error_msg}")

                    # 文件日志输出
                    if actual_log_file:
                        _write_log(actual_log_file, detailed_error_msg, "ERROR")

                    # 如果还有重试次数
                    if attempt <= max_retries:
                        # 调用重试回调函数
                        if on_retry:
                            try:
                                on_retry(attempt, e, retry_delay)
                            except Exception:
                                pass

                        # 输出重试等待信息
                        if print_error:
                            print(f"等待 {retry_delay} 秒后重试... (剩余重试次数: {max_retries - attempt + 1})")

                        # 等待延迟
                        time.sleep(retry_delay)
                    else:
                        # 重试次数用尽
                        break

            # 所有尝试都失败了
            if last_exception is not None:
                # 计算总执行时间
                end_time = datetime.now()
                total_execution_time = (end_time - start_time).total_seconds()

                # 构造最终失败信息
                final_error_msg = (f"函数 {function_name} 所有 {max_retries + 1} 次尝试均失败 - "
                                   f"总耗时: {total_execution_time:.2f}秒")

                if print_error:
                    print(f"{'=' * 60}")
                    print(f"{final_error_msg}")
                    print(f"{'=' * 60}")

                if actual_log_file:
                    _write_log(actual_log_file, final_error_msg, "CRITICAL")

                # 异常处理策略
                if raise_exception:
                    raise last_exception
                elif continue_on_error:
                    if print_error:
                        print(f"继续执行（忽略 {function_name} 的错误）")
                    return None
                else:
                    # 默认行为：不继续执行，程序会停止
                    if print_error:
                        print(f"函数 {function_name} 执行失败，停止程序")
                    raise last_exception  # 或者 return None，根据你的需求

            # 理论上不会执行到这里
            return None

        return wrapper

    return decorator

def decorator_func_profile(func):
    """
        性能分析装饰器
        用于分析函数的执行时间和性能瓶颈
        输出结果说明：
            ncalls: 函数调用次数
            tottime: 函数本身执行的总时间（不包括子函数）
            percall: 每次调用的平均时间 (tottime/ncalls)
            cumtime: 累计时间（包括子函数调用）
            percall: 每次调用的累计平均时间
            filename:lineno(function): 文件名、行号和函数名
        Args:
            func: 被装饰的函数
        Returns:
            装饰后的函数，会在执行时输出性能分析报告
        """
    @functools.wraps(func)  # 保持原函数的元数据（名称、文档字符串等）
    def wrapper(*args, **kwargs):
        # 创建性能分析器实例
        profiler = cProfile.Profile()
        # 运行被装饰的函数并进行性能分析
        # runcall方法会执行函数并记录性能数据
        result = profiler.runcall(func, *args, **kwargs)
        # 创建性能统计对象
        stats = pstats.Stats(profiler)
        # 按累计时间排序（从大到小）
        # 'cumulative' 表示包括子函数调用在内的总时间
        # 其他排序选项：'time'（函数自身时间）、'calls'（调用次数）等
        stats.sort_stats('cumulative')
        # 打印性能统计结果，只显示前10个最耗时的函数
        # 这样可以快速定位性能瓶颈
        stats.print_stats(10)
        # 返回原函数的执行结果，确保装饰器不影响函数正常功能
        return result
    return wrapper

def interactive_method_explorer(target):
    """交互式方法探索器 - 支持类和实例
    1. 基础命令
        q - 退出探索器

        list - 重新显示所有方法列表

    2. 方法查询命令
        方法名 - 直接输入方法名查看详细信息

        例如：simple_method、__init__、class_method

    3. 方法测试命令（仅适用于实例）
        test 方法名 - 测试调用实例的方法

        例如：test simple_method、test method_with_args
"""

    def test_call_method(instance, method_name, method):
        """测试调用实例的方法"""
        print(f"\n🧪 测试调用: {method_name}")
        print("=" * 40)

        try:
            # 获取方法签名来分析参数
            sig = inspect.signature(method)
            params = list(sig.parameters.values())

            # 移除第一个参数（通常是self）
            if params and params[0].name in ['self', 'cls']:
                params = params[1:]

            if not params:
                # 无参数方法
                print("🔹 这是一个无参数方法")
                result = getattr(instance, method_name)()
                print(f"✅ 调用结果: {result}")
            else:
                # 有参数方法，提示用户
                print(f"🔹 这个方法需要 {len(params)} 个参数:")
                for param in params:
                    default = f" (默认值: {param.default})" if param.default != param.empty else ""
                    print(
                        f"   - {param.name}: {param.annotation if param.annotation != param.empty else '任意类型'}{default}")

                print("❌ 自动调用有参数的方法有风险，建议手动调用")
                print("💡 你可以这样手动调用:")
                print(f"   result = your_instance.{method_name}(参数1, 参数2, ...)")

        except Exception as e:
            print(f"❌ 调用失败: {e}")

    # 判断输入是类还是实例
    if isinstance(target, type):
        cls = target
        target_name = cls.__name__
        target_type = "类"
    else:
        cls = type(target)
        target_name = f"{cls.__name__} 实例"
        target_type = "实例"

    print(f"🔍 探索 {target_type}: {target_name}")
    print("=" * 50)

    # 收集方法
    methods_dict = {}
    for name, method in inspect.getmembers(cls):
        if callable(method):
            methods_dict[name] = method

    print(f"可用的方法 ({len(methods_dict)} 个):")
    for i, name in enumerate(sorted(methods_dict.keys()), 1):
        # 显示方法类型标识
        method = methods_dict[name]
        if isinstance(method, classmethod):
            type_flag = "[类方法]"
        elif isinstance(method, staticmethod):
            type_flag = "[静态方法]"
        elif name.startswith('__') and name.endswith('__'):
            type_flag = "[特殊方法]"
        elif name.startswith('_'):
            type_flag = "[受保护]"
        else:
            type_flag = "[实例方法]"

        print(f"{i:2d}. {name} {type_flag}")

    print("\n提示:")
    print("- 输入方法名查看详细信息")
    print("- 输入 'list' 重新显示方法列表")
    print("- 输入 'test 方法名' 测试调用方法（仅实例）")
    print("- 输入 'q' 退出")
    print("=" * 50)

    while True:
        try:
            choice = input("\n>>> ").strip()

            if choice.lower() == 'q':
                print("退出探索器")
                break

            elif choice.lower() == 'list':
                # 重新显示列表
                print(f"可用的方法 ({len(methods_dict)} 个):")
                for i, name in enumerate(sorted(methods_dict.keys()), 1):
                    method = methods_dict[name]
                    if isinstance(method, classmethod):
                        type_flag = "[类方法]"
                    elif isinstance(method, staticmethod):
                        type_flag = "[静态方法]"
                    elif name.startswith('__') and name.endswith('__'):
                        type_flag = "[特殊方法]"
                    elif name.startswith('_'):
                        type_flag = "[受保护]"
                    else:
                        type_flag = "[实例方法]"
                    print(f"{i:2d}. {name} {type_flag}")
                continue

            elif choice.startswith('test '):
                # 测试调用方法
                if not isinstance(target, type):  # 只有实例可以测试调用
                    method_name = choice[5:].strip()
                    if method_name in methods_dict:
                        test_call_method(target, method_name, methods_dict[method_name])
                    else:
                        print(f"方法 '{method_name}' 不存在")
                else:
                    print("❌ 测试调用仅适用于实例，不适用于类")
                continue

            # 普通方法查询
            if choice in methods_dict:
                method = methods_dict[choice]
                print(f"\n{'=' * 60}")
                print(f"📖 方法 '{choice}' 的详细信息:")
                print(f"{'=' * 60}")

                # 方法类型
                if isinstance(method, classmethod):
                    print("📝 类型: 类方法")
                elif isinstance(method, staticmethod):
                    print("📝 类型: 静态方法")
                elif choice.startswith('__') and choice.endswith('__'):
                    print("📝 类型: 特殊方法")
                elif choice.startswith('_'):
                    print("📝 类型: 受保护/私有方法")
                else:
                    print("📝 类型: 实例方法")

                # 方法签名
                try:
                    sig = inspect.signature(method)
                    print(f"✍️  签名: {sig}")

                    # 参数详情
                    if sig.parameters:
                        print("📋 参数详情:")
                        for param_name, param in sig.parameters.items():
                            default = param.default if param.default != param.empty else "无默认值"
                            annotation = param.annotation if param.annotation != param.empty else "无类型提示"
                            kind = param.kind
                            print(f"    - {param_name}: 默认值={default}, 类型={kind}, 类型提示={annotation}")
                except (ValueError, TypeError) as e:
                    print(f"❌ 签名错误: {e}")

                # 文档字符串
                doc = inspect.getdoc(method)
                if doc:
                    print(f"📄 文档:\n{doc}")
                else:
                    print("📄 文档: 无文档字符串")

                # 源代码
                try:
                    source = inspect.getsource(method)
                    print(f"📜 源代码:\n{source}")
                except (OSError, TypeError):
                    print("📜 源代码: 无法获取（可能是C扩展或内置方法）")

                # 如果是实例，显示调用提示
                if not isinstance(target, type) and not choice.startswith('_'):
                    print(f"💡 提示: 可以使用 'test {choice}' 来测试调用此方法")

            else:
                print("❌ 方法不存在，请重试")
                print("💡 提示: 输入 'list' 查看所有可用方法")

        except KeyboardInterrupt:
            print("\n退出探索器")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

def detailed_method_analysis(obj, log_file_path=None):
    """详细的方法分析 - 支持类和实例 - 结果输出到日志文件
    obj可以是方法或者实例，log_file_path是输出路径
    默认是输出到桌面的 方法分析.log文件
    """
    # 设置默认日志文件路径（桌面）
    if log_file_path is None:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        log_file_path = os.path.join(desktop, "方法分析.log")
    def write_method_details_advanced(file, name, method, cls):
        """将方法的进阶详细信息写入文件"""
        file.write(f"\n🎯 {name}:\n")
        # 方法类型
        if isinstance(method, classmethod):
            file.write("  📝 类型: 类方法\n")
        elif isinstance(method, staticmethod):
            file.write("  📝 类型: 静态方法\n")
        elif name.startswith('__') and name.endswith('__'):
            file.write("  📝 类型: 特殊方法\n")
        elif name.startswith('_'):
            if name.startswith('__'):
                file.write("  📝 类型: 私有方法\n")
            else:
                file.write("  📝 类型: 受保护方法\n")
        else:
            file.write("  📝 类型: 实例方法\n")

        # 文档字符串
        doc = inspect.getdoc(method)
        if doc:
            file.write(f"  📄 文档:\n    {doc}\n")

        # 方法签名
        try:
            sig = inspect.signature(method)
            file.write(f"  ✍️ 签名: {sig}\n")
            # 参数详情
            if sig.parameters:
                file.write("  📋 参数详情:\n")
                for param_name, param in sig.parameters.items():
                    default = param.default if param.default != param.empty else "无默认值"
                    annotation = param.annotation if param.annotation != param.empty else "无类型提示"
                    file.write(f"    - {param_name}: 默认值={default}, 类型提示={annotation}\n")
        except (ValueError, TypeError) as e:
            file.write(f"  ❌ 签名错误: {e}\n")

        # 源代码行数
        try:
            lines, start_line = inspect.getsourcelines(method)
            file.write(f"  📏 代码位置: 第{start_line}行, 共{len(lines)}行\n")
        except (OSError, TypeError):
            pass

    # 打开日志文件进行写入
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        if isinstance(obj, type):
            # 如果是类
            cls = obj
            cls_name = cls.__name__
        else:
            # 如果是实例
            cls = type(obj)
            cls_name = f"{cls.__name__} 实例"

        log_file.write(f"📊 详细分析: {cls_name}\n\n")

        methods = inspect.getmembers(cls, predicate=inspect.ismethod)
        functions = inspect.getmembers(cls, predicate=inspect.isfunction)

        log_file.write("=== 实例方法 ===\n")
        for name, method in functions:
            if not name.startswith('_') or (name.startswith('__') and name.endswith('__')):
                write_method_details_advanced(log_file, name, method, cls)

        log_file.write("\n=== 类方法和静态方法 ===\n")
        for name, method in methods:
            if not name.startswith('_'):
                write_method_details_advanced(log_file, name, method, cls)

        log_file.write("\n=== 受保护和私有方法 ===\n")
        for name, method in functions + methods:
            if name.startswith('_') and not (name.startswith('__') and name.endswith('__')):
                write_method_details_advanced(log_file, name, method, cls)

        log_file.write("\n=== 特殊方法 (魔术方法) ===\n")
        for name, method in functions + methods:
            if name.startswith('__') and name.endswith('__'):
                write_method_details_advanced(log_file, name, method, cls)

    print(f"✅ 分析完成！结果已保存到: {log_file_path}")
    return log_file_path

def get_class_mro(obj):
    """快速查看类信息"""
    print(f"实例: {obj}")
    print(f"类型: {type(obj)}")
    print(f"所属类: {obj.__class__}")
    print("继承链:", " -> ".join([cls.__name__ for cls in obj.__class__.__mro__]))

def detailed_class_info(cls_or_obj, output_to_file=True, log_file_path=None):
    """提供详细的类信息
    Args:
        cls_or_obj: 要分析的类或实例
        output_to_file: 是否输出到日志，默认为True
        log_file_path: 日志文件路径，默认为桌面的"属性和方法.log"
    """
    if isinstance(cls_or_obj, type):
        cls = cls_or_obj
        obj = cls()
    else:
        cls = type(cls_or_obj)
        obj = cls_or_obj

    def output_content(content):
        """根据设置输出内容到文件或控制台"""
        if output_to_file:
            log_file.write(content + '\n')
        else:
            print(content)

    # 如果输出到文件，设置文件路径
    if output_to_file:
        if log_file_path is None:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            log_file_path = os.path.join(desktop, "属性和方法.log")

        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            output_content(f"类名: {cls.__name__}")

            output_content("\n=== 实例属性 ===")
            for key, value in vars(obj).items():
                output_content(f"  {key}:---- \n{value} ({type(value)})")

            output_content("\n=== 方法 ===")
            for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
                if not name.startswith('__'):
                    output_content(f"  方法: {name}")

            output_content("\n=== 函数方法 ===")
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if not name.startswith('__'):
                    output_content(f"  函数: {name}")

            output_content("\n=== 特殊方法 ===")
            for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
                if name.startswith('__') and name.endswith('__'):
                    output_content(f"  特殊方法: {name}")

        print(f"✅ 类信息分析完成！结果已保存到: {log_file_path}")
        return log_file_path

    else:
        # 直接打印到控制台
        print(f"类名: {cls.__name__}")

        print("\n=== 实例属性 ===")
        for key, value in vars(obj).items():
            print(f"  {key}: {value} ({type(value)})")

        print("\n=== 方法 ===")
        for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
            if not name.startswith('__'):
                print(f"  方法: {name}")

        print("\n=== 函数方法 ===")
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith('__'):
                print(f"  函数: {name}")

        print("\n=== 特殊方法 ===")
        for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
            if name.startswith('__') and name.endswith('__'):
                print(f"  特殊方法: {name}")

        return None


def main():
    return

if __name__ == "__main__":
    main()