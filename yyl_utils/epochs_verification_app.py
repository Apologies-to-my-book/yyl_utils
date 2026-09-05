"""Epoch 波形核验 App。

文件按“配置与验证、HDF5 存储、绘图函数、Qt 窗口、公开启动入口”的顺序组织。
公开入口先在 working folder 中解析三个固定文件，再把轻量级属性和分页状态交给
Qt；waveform 与 continuous 大数组始终通过 HDF5 Dataset 按需读取。

本模块生成的 HDF5 文件结构
============================

一、``epochs.h5``：不等长 epoch 波形和属性

文件根节点保存以下 attributes：

* ``schema_version``：整数 schema 版本号；
* ``store_type``：UTF-8 字符串 ``"epochs"``；
* ``has_start_timestamps``：布尔值，表示是否写入 ``/start_timestamps``。

根节点下的固定路径和数据含义如下（设 epoch 数为 ``E``，第 ``i`` 条波形长度
为 ``N_i``）：

``/waveforms``（Group）
    ``values``（一维、分块 Dataset）：把所有 waveform 按 epoch 顺序依次写入的
    扁平数值数组，长度为 ``sum(N_i)``。读取单条波形时只切片该 Dataset 的一个
    区间，不会把全部波形拼成一个新的 ndarray。

    ``offsets``（一维 int64 Dataset，形状 ``(E + 1,)``）：``offsets[i]`` 是第
    ``i`` 条波形在 ``values`` 中的起点，``offsets[i + 1]`` 是终点；因此第 ``i``
    条波形对应 ``values[offsets[i]:offsets[i + 1]]``。

``/start_timestamps``（可选的一维实数 Dataset，形状 ``(E,)``）
    第 ``i`` 条 epoch 的真实开始时间。不存在时，Detail 窗口不能依据 epoch
    开始时间定位连续数据。

``/epochs_ids``（一维 UTF-8 字符串 Dataset，形状 ``(E,)``）
    每条 epoch 的唯一业务 ID；数组下标与 ``waveforms``、``offsets``、属性数组
    的第一个维度一一对应。

``/properties``（Group）
    Group 内每个 ``<property_name>`` 都是一维 bool Dataset，形状为 ``(E,)``。
    ``/properties/is_delete`` 是必需属性；其它属性由输入数据动态决定。属性数值
    在内存中可局部修改，并按连续下标区间写回原 Dataset。

二、``continuous.h5``：连续信号、通道和曲线

文件根节点保存 ``schema_version``（整数）和 ``store_type``（UTF-8 字符串
``"continuous"``）两个 attributes。

``/common_timestamps``（可选的一维实数、分块 Dataset，形状 ``(T,)``）
    所有未提供独立时间轴的曲线共用的时间戳，只保存一份。该数组必须单调非
    递减；没有公共时间轴时，每条曲线都必须提供自己的 ``timestamps``。

``/channels``（Group）
    通道使用稳定的内部编号 Group 保存，编号格式为
    ``channel_000000``、``channel_000001``……。每个通道 Group 包含：

    * ``name`` attribute：UTF-8 原始通道名称；
    * ``/channels/channel_xxxxxx/curves``（Group）：该通道的曲线集合。

``/channels/channel_xxxxxx/curves/curve_yyyyyy``（Group）
    曲线使用通道内稳定编号保存，编号格式为 ``curve_000000``……。每个曲线
    Group 包含：

    * ``name`` attribute：UTF-8 原始曲线名称；
    * ``timestamp_source`` attribute：UTF-8 字符串 ``"common"`` 或 ``"own"``；
    * ``values``（一维、分块数值 Dataset，形状 ``(M,)``）；
    * 当 ``timestamp_source == "own"`` 时，额外包含 ``timestamps``（一维、分块
      实数 Dataset，形状 ``(M,)``）；使用公共时间轴时不重复保存该 Dataset。

所有曲线的 ``values`` 与其实际时间轴长度必须相同。Store 通过 Dataset 切片和
时间戳二分查找按需读取窗口数据；创建文件时先写入同目录临时文件，刷新并关闭
后再原子替换目标文件。
"""

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import json
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import h5py
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QLabel,
                             QScrollArea, QFrame, QMessageBox, QComboBox,
                             QLineEdit, QSizePolicy, QGroupBox, QSlider,
                             QDialog, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==================== 默认配置和输入验证 ====================

DEFAULT_CONFIG = {
    "overview": {
        "rows_per_page": 5,
        "cols_per_page": 8,
        "window_width": 1800,
        "window_height": 1200,
        "subplot_min_height": 160,
        "xlim": None,
        "ylim": None,
        "line_color": "#2563eb",
        "line_width": 0.8,
        "figure_dpi": 90,
    },
    "detail": {
        "subplot_height": 250,
        "figure_width": 1400,
        "window_width": 1500,
        "window_height": 900,
        "time_before": 5.0,
        "time_after": 5.0,
        "timestamp_unit": "s",
        "ylim": [-0.5, 0.5],
        "channel_ylims": {},
        "plot_visibility": {
            "channels": {},
            "curves": {},
        },
        "line_width": 0.8,
        "show_legend": True,
        "max_points_per_pixel": 2,
        "slider_points_per_second": 3000,
    },
    "filters": {
        "default_state": "all",
        "property_states": {
            "is_delete": "false",
        },
    },
    "storage": {
        "overwrite_existing": False,
        "waveforms_compression": "gzip",
        "waveforms_compression_level": 1,
        "waveforms_chunk_points": 1000000,
        "continuous_compression": None,
        "continuous_compression_level": None,
        "continuous_chunk_points": 262144,
        "properties_chunk_size": 65536,
    },
    "performance": {
        "detail_slice_cache_size": 32,
        "show_import_progress": True,
    },
}


_VALID_FILTER_STATES = frozenset({"all", "true", "false"})
_COMMON_TIMESTAMPS_KEY = "__common_timestamps__"
_DEFAULT_VALIDATION_CHUNK_SIZE = 262144
_epochs_SCHEMA_VERSION = 1
_CONTINUOUS_SCHEMA_VERSION = 1


# 全局视觉主题：深蓝色用于主操作，青绿色用于辅助操作，橙红色只表示警告或删除。
# 该样式同时覆盖主窗口、Detail 窗口、独立设置窗口及其常用输入控件。
APP_STYLE_SHEET = """
QMainWindow, QDialog {
    background-color: #eef3f8;
    color: #1e293b;
}
QWidget {
    color: #1e293b;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}
QPushButton {
    background-color: #29466f;
    color: #ffffff;
    border: 1px solid #29466f;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #35639a;
    border-color: #35639a;
}
QPushButton:pressed {
    background-color: #1d3557;
    border-color: #1d3557;
}
QPushButton:disabled {
    background-color: #b8c4d2;
    border-color: #b8c4d2;
    color: #edf2f7;
}
QPushButton#settingsToggleButton, QPushButton#saveSettingsButton,
QPushButton#curveSettingsButton {
    background-color: #0f766e;
    border-color: #0f766e;
}
QPushButton#settingsToggleButton:hover, QPushButton#saveSettingsButton:hover,
QPushButton#curveSettingsButton:hover {
    background-color: #15998f;
    border-color: #15998f;
}
QLineEdit, QComboBox {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #b9c7d8;
    border-radius: 5px;
    padding: 5px 8px;
    selection-background-color: #3b82f6;
}
QLineEdit:focus, QComboBox:focus {
    border: 2px solid #3b82f6;
    padding: 4px 7px;
}
QComboBox::drop-down {
    width: 24px;
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #1e293b;
    selection-background-color: #dbeafe;
    selection-color: #1e3a8a;
}
QCheckBox {
    spacing: 8px;
    color: #29466f;
    padding: 4px 2px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #9db1c9;
    border-radius: 4px;
    background: #ffffff;
}
QCheckBox::indicator:hover {
    border-color: #3b82f6;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border-color: #2563eb;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #cbd8e6;
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
    color: #29466f;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    background-color: #ffffff;
}
QScrollArea {
    background-color: #ffffff;
    border: 1px solid #d7e1ec;
    border-radius: 6px;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #e8eef5;
    border: none;
    margin: 0;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #9db1c9;
    border-radius: 5px;
    min-height: 24px;
    min-width: 24px;
}
QScrollBar::handle:hover {
    background: #6f8eaf;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #cbd8e6;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #3b82f6;
    border-radius: 3px;
}
QSlider::add-page:horizontal {
    background: #dbe5f0;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: #2563eb;
    border: 2px solid #ffffff;
}
QToolBar {
    background-color: #e3ebf4;
    border: 1px solid #cbd8e6;
    spacing: 4px;
}
QLabel#infoLabel {
    color: #29466f;
    padding: 5px 7px;
}
"""


def merge_config_dict(config_dict=None):
    """将用户配置递归合并到默认配置中，并返回独立的新字典。

    与普通的 ``dict.update()`` 不同，本函数会递归处理嵌套字典。因此用户只需
    提供真正想修改的字段，其余字段仍保留 ``DEFAULT_CONFIG`` 中的默认值。
    函数会深复制默认配置和用户值，调用方后续修改返回结果时不会意外修改
    ``DEFAULT_CONFIG`` 或原始 ``config_dict``。

    Parameters
    ----------
    config_dict : Mapping or None, optional
        用户提供的局部配置。传入 ``None`` 时返回默认配置的深复制。支持的完整
        配置来源、合并规则和文件冲突行为详见 ``DataViewer.__init__`` 的
        ``config_dict`` 参数。

    Returns
    -------
    dict
        合并后的完整配置副本。

    Raises
    ------
    ValueError
        ``config_dict`` 不是字典或其他 Mapping 对象时抛出。

    Workflow
    --------
    1. 深复制 ``DEFAULT_CONFIG``，保证模块常量不被调用方间接修改。
    2. 递归遍历用户字典；遇到两个 Mapping 时继续向下合并，否则用用户值替换。
    3. 返回与默认配置、用户配置都不共享可变子对象的新字典。
    """
    if config_dict is None:
        return deepcopy(DEFAULT_CONFIG)
    if not isinstance(config_dict, Mapping):
        raise ValueError("config_dict 必须是字典或其他 Mapping 对象。")

    def merge_mapping(base_mapping, override_mapping):
        """递归合并两个 Mapping；输入和输出都是键值映射，返回独立 dict。

        Parameters
        ----------
        base_mapping : Mapping[str, object]
            当前层的默认键值映射。
        override_mapping : Mapping[str, object]
            当前层的用户覆盖映射。

        Returns
        -------
        dict[str, object]
            完成当前层及所有嵌套层合并后的深复制字典。

        Workflow
        --------
        先复制默认层，再逐键判断是否需要递归；非 Mapping 值直接深复制覆盖。
        """
        merged = deepcopy(dict(base_mapping))
        for key, override_value in override_mapping.items():
            base_value = merged.get(key)
            if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
                merged[key] = merge_mapping(base_value, override_value)
            else:
                merged[key] = deepcopy(override_value)
        return merged

    return merge_mapping(DEFAULT_CONFIG, config_dict)


def _is_bool_scalar(value):
    """判断单个对象是否为 Python 或 NumPy 布尔标量。

    Parameters
    ----------
    value : object
        任意待检查对象；数组和整数不会被视为布尔标量。

    Returns
    -------
    bool
        ``value`` 为 ``bool`` 或 ``numpy.bool_`` 时返回 True，否则返回 False。

    Workflow
    --------
    使用 ``isinstance`` 同时检查两种布尔标量类型，不读取容器内容。
    """
    return isinstance(value, (bool, np.bool_))


def _is_integer_scalar(value):
    """判断对象是否为非布尔整数标量。

    Parameters
    ----------
    value : object
        任意待检查对象，支持 Python/NumPy 的 Integral 标量。

    Returns
    -------
    bool
        是整数且不是布尔值时为 True。

    Workflow
    --------
    先按 ``numbers.Integral`` 判断整数，再显式排除其子类 ``bool``。
    """
    return isinstance(value, Integral) and not _is_bool_scalar(value)


def _is_real_scalar(value):
    """判断对象是否为有限、非布尔的实数标量。

    Parameters
    ----------
    value : object
        任意待检查对象，支持 Python 和 NumPy 实数标量。

    Returns
    -------
    bool
        为实数、不是布尔值且不是 NaN/正负无穷时返回 True。

    Workflow
    --------
    依次检查 ``Real`` 类型、排除 bool，并用 ``numpy.isfinite`` 检查有限性。
    """
    return (
        isinstance(value, Real)
        and not _is_bool_scalar(value)
        and bool(np.isfinite(value))
    )


def _require_config_section(config_dict, section_name):
    """取得必需配置区块，并生成带字段路径的中文错误。

    Parameters
    ----------
    config_dict : Mapping[str, object]
        完整配置字典。配置来源和默认结构详见 ``DataViewer.__init__`` 的
        ``config_dict`` 参数。
    section_name : str
        顶层区块名称，例如 ``"overview"`` 或 ``"detail"``。

    Returns
    -------
    Mapping[str, object]
        指定配置区块的原引用，不复制其中内容。

    Workflow
    --------
    先检查顶层键是否存在，再检查对应值是否为 Mapping，最后返回该区块。

    Raises
    ------
    ValueError
        区块缺失或值不是 Mapping 时抛出。
    """
    if section_name not in config_dict:
        raise ValueError(f"config_dict 缺少必需区块 '{section_name}'。")
    section = config_dict[section_name]
    if not isinstance(section, Mapping):
        raise ValueError(f"config_dict['{section_name}'] 必须是字典。")
    return section


def _require_config_value(section, section_name, key):
    """从一个配置区块取得必需字段。

    Parameters
    ----------
    section : Mapping[str, object]
        某个配置区块。
    section_name : str
        区块名称，仅用于错误信息。
    key : str
        需要读取的字段名称。

    Returns
    -------
    object
        ``section[key]`` 的原值，具体类型由字段定义决定。

    Workflow
    --------
    检查键是否存在；存在则直接返回，不执行类型转换。

    Raises
    ------
    ValueError
        字段缺失时抛出，避免向用户暴露难定位的 KeyError。
    """
    if key not in section:
        raise ValueError(f"config_dict['{section_name}'] 缺少字段 '{key}'。")
    return section[key]


def _validate_positive_integer(value, field_path):
    """验证配置值是大于 0 的非布尔整数。

    Parameters
    ----------
    value : object
        待检查的配置值。
    field_path : str
        错误信息显示的完整配置路径。

    Returns
    -------
    None
        验证成功时不返回数据。

    Workflow
    --------
    调用整数标量判断，再检查数值下界；失败时抛出带路径的 ValueError。
    """
    if not _is_integer_scalar(value) or value <= 0:
        raise ValueError(f"{field_path} 必须是正整数，当前值为 {value!r}。")


def _validate_non_negative_integer(value, field_path):
    """验证配置值是大于等于 0 的非布尔整数。

    Parameters
    ----------
    value : object
        待检查的配置值。
    field_path : str
        错误信息显示的完整配置路径。

    Returns
    -------
    None
        验证成功时不返回数据。

    Workflow
    --------
    检查整数类型与非负下界，不修改传入值。
    """
    if not _is_integer_scalar(value) or value < 0:
        raise ValueError(f"{field_path} 必须是非负整数，当前值为 {value!r}。")


def _validate_positive_number(value, field_path):
    """验证配置值是大于 0 的有限实数。

    Parameters
    ----------
    value : object
        待检查的数值标量。
    field_path : str
        错误信息显示的完整配置路径。

    Returns
    -------
    None
        验证成功时不返回数据。

    Workflow
    --------
    先排除 bool、NaN 和无穷，再检查严格大于 0。
    """
    if not _is_real_scalar(value) or value <= 0:
        raise ValueError(f"{field_path} 必须是大于 0 的有限数值，当前值为 {value!r}。")


def _validate_non_negative_number(value, field_path):
    """验证配置值是大于等于 0 的有限实数。

    Parameters
    ----------
    value : object
        待检查的数值标量。
    field_path : str
        错误信息显示的完整配置路径。

    Returns
    -------
    None
        验证成功时不返回数据。

    Workflow
    --------
    先检查有限实数类型，再检查非负下界。
    """
    if not _is_real_scalar(value) or value < 0:
        raise ValueError(f"{field_path} 必须是大于等于 0 的有限数值，当前值为 {value!r}。")


def _validate_range_setting(value, field_path):
    """验证 xlim、ylim 等二元素数值范围。

    Parameters
    ----------
    value : None or Sequence[Real]
        ``None`` 表示自动范围；否则必须是长度为 2 的有限实数序列
        ``[lower, upper]``。
    field_path : str
        错误信息显示的完整配置路径。

    Returns
    -------
    None
        范围合法时不返回数据。

    Workflow
    --------
    允许 None；否则依次验证非字符串序列、长度、两个端点类型以及 lower < upper。

    Raises
    ------
    ValueError
        形状、类型、有限性或端点顺序不正确时抛出。
    """
    if value is None:
        return
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field_path} 必须是 None 或包含两个数值的序列。")
    try:
        if len(value) != 2:
            raise ValueError(f"{field_path} 必须恰好包含两个数值。")
        lower, upper = value
    except TypeError as exc:
        raise ValueError(f"{field_path} 必须是 None 或包含两个数值的序列。") from exc
    if not _is_real_scalar(lower) or not _is_real_scalar(upper):
        raise ValueError(f"{field_path} 的最小值和最大值都必须是有限数值。")
    if lower >= upper:
        raise ValueError(
            f"{field_path} 必须满足最小值小于最大值，当前范围为 {value!r}。"
        )


def validate_config_dict(config_dict):
    """验证已经与默认值合并后的完整配置。

    本函数只检查配置，不修改输入。推荐的调用顺序是先执行
    ``merge_config_dict(user_config)``，再将返回值传给本函数。这样既允许用户只
    提供局部配置，又能保证 App 后续读取到的所有必需字段都存在且类型正确。

    Parameters
    ----------
    config_dict : Mapping
        需要验证的完整配置。通常应先由 ``merge_config_dict()`` 生成；默认配置
        来源及调用方式详见 ``DataViewer.__init__`` 的 ``config_dict`` 参数。

    Returns
    -------
    None
        验证成功时不返回新对象。

    Raises
    ------
    ValueError
        配置缺少必需字段，或任一字段类型、范围不符合要求时抛出。

    Workflow
    --------
    1. 取得 overview、detail、filters、storage、performance 五个必需区块。
    2. 按字段语义检查整数、实数、范围、布尔值、筛选枚举和嵌套字典。
    3. 所有检查均只读；成功后返回 None，调用方继续使用原配置对象。
    """
    if not isinstance(config_dict, Mapping):
        raise ValueError("config_dict 必须是字典或其他 Mapping 对象。")

    overview = _require_config_section(config_dict, "overview")
    detail = _require_config_section(config_dict, "detail")
    filters = _require_config_section(config_dict, "filters")
    storage = _require_config_section(config_dict, "storage")
    performance = _require_config_section(config_dict, "performance")

    for key in ("rows_per_page", "cols_per_page", "window_width", "window_height"):
        value = _require_config_value(overview, "overview", key)
        _validate_positive_integer(value, f"config_dict['overview']['{key}']")
    _validate_positive_number(
        _require_config_value(overview, "overview", "subplot_min_height"),
        "config_dict['overview']['subplot_min_height']",
    )
    _validate_range_setting(
        _require_config_value(overview, "overview", "xlim"),
        "config_dict['overview']['xlim']",
    )
    _validate_range_setting(
        _require_config_value(overview, "overview", "ylim"),
        "config_dict['overview']['ylim']",
    )
    line_color = _require_config_value(overview, "overview", "line_color")
    if not isinstance(line_color, str) or not line_color:
        raise ValueError("config_dict['overview']['line_color'] 必须是非空字符串。")
    _validate_positive_number(
        _require_config_value(overview, "overview", "line_width"),
        "config_dict['overview']['line_width']",
    )
    _validate_positive_number(
        _require_config_value(overview, "overview", "figure_dpi"),
        "config_dict['overview']['figure_dpi']",
    )

    for key in ("window_width", "window_height"):
        value = _require_config_value(detail, "detail", key)
        _validate_positive_integer(value, f"config_dict['detail']['{key}']")
    for key in (
            "subplot_height",
            "figure_width",
            "line_width",
            "max_points_per_pixel",
            "slider_points_per_second",
    ):
        value = _require_config_value(detail, "detail", key)
        _validate_positive_number(value, f"config_dict['detail']['{key}']")

    time_before = _require_config_value(detail, "detail", "time_before")
    time_after = _require_config_value(detail, "detail", "time_after")
    _validate_non_negative_number(time_before, "config_dict['detail']['time_before']")
    _validate_non_negative_number(time_after, "config_dict['detail']['time_after']")
    if time_before == 0 and time_after == 0:
        raise ValueError("time_before 和 time_after 不能同时为 0。")

    timestamp_unit = _require_config_value(detail, "detail", "timestamp_unit")
    if not isinstance(timestamp_unit, str) or not timestamp_unit.strip():
        raise ValueError("config_dict['detail']['timestamp_unit'] 必须是非空字符串。")
    _validate_range_setting(
        _require_config_value(detail, "detail", "ylim"),
        "config_dict['detail']['ylim']",
    )

    channel_ylims = _require_config_value(detail, "detail", "channel_ylims")
    if not isinstance(channel_ylims, Mapping):
        raise ValueError("config_dict['detail']['channel_ylims'] 必须是字典。")
    for channel_name, channel_ylim in channel_ylims.items():
        _validate_range_setting(
            channel_ylim,
            f"config_dict['detail']['channel_ylims'][{channel_name!r}]",
        )

    plot_visibility = detail.get("plot_visibility", {})
    if not isinstance(plot_visibility, Mapping):
        raise ValueError("config_dict['detail']['plot_visibility'] 必须是字典。")
    visibility_channels = plot_visibility.get("channels", {})
    visibility_curves = plot_visibility.get("curves", {})
    if not isinstance(visibility_channels, Mapping):
        raise ValueError(
            "config_dict['detail']['plot_visibility']['channels'] 必须是字典。"
        )
    if not isinstance(visibility_curves, Mapping):
        raise ValueError(
            "config_dict['detail']['plot_visibility']['curves'] 必须是字典。"
        )
    for channel_name, enabled in visibility_channels.items():
        if not isinstance(channel_name, str) or not _is_bool_scalar(enabled):
            raise ValueError("plot_visibility.channels 的值必须是布尔值。")
    for channel_name, curve_mapping in visibility_curves.items():
        if not isinstance(channel_name, str) or not isinstance(curve_mapping, Mapping):
            raise ValueError("plot_visibility.curves 必须是通道到曲线字典的映射。")
        for curve_name, enabled in curve_mapping.items():
            if not isinstance(curve_name, str) or not _is_bool_scalar(enabled):
                raise ValueError("plot_visibility.curves 的曲线值必须是布尔值。")

    show_legend = _require_config_value(detail, "detail", "show_legend")
    if not _is_bool_scalar(show_legend):
        raise ValueError("config_dict['detail']['show_legend'] 必须是布尔值。")

    default_filter_state = _require_config_value(filters, "filters", "default_state")
    if (
            not isinstance(default_filter_state, str)
            or default_filter_state not in _VALID_FILTER_STATES
    ):
        raise ValueError(
            "config_dict['filters']['default_state'] 只能是 'all'、'true' 或 'false'。"
        )
    property_states = _require_config_value(filters, "filters", "property_states")
    if not isinstance(property_states, Mapping):
        raise ValueError("config_dict['filters']['property_states'] 必须是字典。")
    for property_name, state in property_states.items():
        if not isinstance(state, str) or state not in _VALID_FILTER_STATES:
            raise ValueError(
                f"属性 {property_name!r} 的筛选状态只能是 'all'、'true' 或 'false'，"
                f"当前值为 {state!r}。"
            )

    overwrite_existing = _require_config_value(storage, "storage", "overwrite_existing")
    if not _is_bool_scalar(overwrite_existing):
        raise ValueError("config_dict['storage']['overwrite_existing'] 必须是布尔值。")
    for key in ("waveforms_chunk_points", "continuous_chunk_points", "properties_chunk_size"):
        value = _require_config_value(storage, "storage", key)
        _validate_positive_integer(value, f"config_dict['storage']['{key}']")

    for key in ("waveforms_compression_level", "continuous_compression_level"):
        value = _require_config_value(storage, "storage", key)
        if value is not None:
            _validate_non_negative_integer(value, f"config_dict['storage']['{key}']")

    _require_config_value(storage, "storage", "waveforms_compression")
    _require_config_value(storage, "storage", "continuous_compression")

    cache_size = _require_config_value(performance, "performance", "detail_slice_cache_size")
    _validate_non_negative_integer(
        cache_size,
        "config_dict['performance']['detail_slice_cache_size']",
    )
    show_progress = _require_config_value(performance, "performance", "show_import_progress")
    if not _is_bool_scalar(show_progress):
        raise ValueError("config_dict['performance']['show_import_progress'] 必须是布尔值。")


def convert_to_json_compatible(value):
    """递归把配置中的 NumPy 对象转换为 JSON 原生类型。

    Parameters
    ----------
    value : object
        任意配置值，可包含 Mapping、list、tuple、NumPy scalar 或 ndarray。

    Returns
    -------
    object
        只由 dict、list、str、int、float、bool 和 None 组成的等价对象。

    Raises
    ------
    TypeError
        遇到 JSON 无法表达的对象时抛出，并保留具体类型名称。

    Workflow
    --------
    1. Mapping 逐键递归转换，并要求所有键都是字符串。
    2. ndarray 先转为嵌套 list，NumPy scalar 先用 ``item()`` 取 Python 标量。
    3. list/tuple 逐元素递归；JSON 原生标量直接返回；其余类型明确报错。
    """
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON 配置的字典键必须是字符串，发现 {key!r}。")
            converted[key] = convert_to_json_compatible(item)
        return converted
    if isinstance(value, np.ndarray):
        return convert_to_json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return convert_to_json_compatible(value.item())
    if isinstance(value, (list, tuple)):
        return [convert_to_json_compatible(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"配置中包含 JSON 不支持的类型：{type(value).__name__}。")


def load_config_json(file_path):
    """以 UTF-8 加载、递归补齐并验证 config.json。

    Parameters
    ----------
    file_path : str or Path
        已存在的 JSON 配置文件。

    Returns
    -------
    dict
        与 ``DEFAULT_CONFIG`` 合并后的完整有效配置。

    Raises
    ------
    ValueError
        文件无法读取、不是 UTF-8、JSON 语法错误、顶层不是 object，或配置验证
        失败时抛出。

    Workflow
    --------
    1. 将路径展开并解析为绝对路径，以 UTF-8 读取 JSON。
    2. 验证顶层是 Mapping，并与默认配置递归合并。
    3. 执行完整配置验证，返回可直接供 App 使用的 dict。
    """
    path = Path(file_path).expanduser().resolve()
    try:
        with path.open("r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取配置文件 {path}：{exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"配置文件 {path} 的顶层必须是 JSON object。")
    effective_config = merge_config_dict(loaded)
    validate_config_dict(effective_config)
    return effective_config


def save_config_json_atomic(file_path, config_dict):
    """使用同目录临时文件和 ``os.replace`` 原子保存 UTF-8 配置。

    写入前会先将 NumPy 标量、布尔值和数组转换为 JSON 原生类型，再执行完整
    配置验证。只有临时文件写入、flush 和 fsync 都成功后才替换目标文件，避免
    程序中断留下半个 JSON。

    Parameters
    ----------
    file_path : str or Path
        最终 config.json 路径，其父目录必须已经存在。
    config_dict : Mapping
        已合并的完整配置。配置来源和结构入口详见 ``DataViewer.__init__`` 的
        ``config_dict`` 参数及模块级 ``DEFAULT_CONFIG``。

    Returns
    -------
    Path
        最终配置文件的绝对路径。

    Raises
    ------
    ValueError
        父目录无效或配置不符合规则时抛出。
    TypeError
        配置包含不能转换为 JSON 的对象时抛出。
    OSError
        临时文件写入、同步或原子替换失败时抛出。

    Workflow
    --------
    1. 解析目标路径，递归转换 NumPy 类型，并再次验证完整配置。
    2. 在目标目录创建临时文件，以 UTF-8、缩进 4 格写入并执行 flush/fsync。
    3. 用 ``os.replace`` 原子替换目标；任何异常都会清理临时文件后重新抛出。
    """
    target_path = Path(file_path).expanduser().resolve()
    if not target_path.parent.exists() or not target_path.parent.is_dir():
        raise ValueError(f"配置目标目录不存在或不是目录：{target_path.parent}")
    compatible_config = convert_to_json_compatible(config_dict)
    validate_config_dict(compatible_config)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as config_file:
            json.dump(
                compatible_config,
                config_file,
                ensure_ascii=False,
                indent=4,
            )
            config_file.write("\n")
            config_file.flush()
            os.fsync(config_file.fileno())
        os.replace(temporary_path, target_path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return target_path


def _resolve_working_folder(working_folder):
    """验证 working folder，并在路径尚不存在时创建它。

    Parameters
    ----------
    working_folder : str or os.PathLike[str]
        App 工作目录路径；不能为 None、HDF5 Store 或已存在的普通文件。

    Returns
    -------
    pathlib.Path
        已存在、已解析为绝对路径的目录对象。

    Workflow
    --------
    依次检查输入、解析绝对路径、排除普通文件，最后递归创建缺失目录。

    Raises
    ------
    ValueError
        路径无效、不是目录或无法创建时抛出。
    """
    if working_folder is None or isinstance(working_folder, EpochsHDF5Store):
        raise ValueError("working_folder 必须是目录路径，不能省略或传入 Store。")
    try:
        folder = Path(working_folder).expanduser().resolve()
    except (TypeError, OSError) as exc:
        raise ValueError(f"working_folder 无法解析：{working_folder!r}。") from exc
    if folder.exists() and not folder.is_dir():
        raise ValueError(f"working_folder 已存在但不是目录：{folder}")
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"无法创建 working_folder {folder}：{exc}") from exc
    return folder


def _passed_config_allows_overwrite(config_dict):
    """判断调用方是否显式传入 ``overwrite_existing=True``。

    Parameters
    ----------
    config_dict : Mapping[str, object] or None
        用户本次调用传入的局部配置，而不是合并后的默认配置。覆盖语义详见
        ``DataViewer.__init__`` 的 ``config_dict`` 参数。

    Returns
    -------
    bool
        仅当 ``storage`` 是 Mapping 且字段值严格为 True 时返回 True。

    Workflow
    --------
    先检查两层 Mapping，再读取布尔开关；缺失或其他值一律按 False 处理。
    """
    if not isinstance(config_dict, Mapping):
        return False
    storage = config_dict.get("storage")
    return isinstance(storage, Mapping) and storage.get("overwrite_existing") is True


def _resolve_config_for_working_folder(working_folder, config_dict):
    """按规范决定有效配置，并处理 config.json 的冲突与原子写入。

    Parameters
    ----------
    working_folder : pathlib.Path
        已验证且存在的工作目录。
    config_dict : Mapping[str, object] or None
        用户局部配置。None 表示优先加载目录中的 config.json；加载、合并与覆盖
        规则详见 ``DataViewer.__init__`` 的 ``config_dict`` 参数。

    Returns
    -------
    dict[str, object]
        与默认值合并并通过验证的完整配置。

    Workflow
    --------
    1. 未传配置时加载已有 JSON，否则使用默认配置。
    2. 已传配置时先合并验证；若 JSON 已存在，要求本次输入显式允许覆盖。
    3. 将传入配置的有效结果原子保存，再返回同一份有效配置。

    Raises
    ------
    FileExistsError
        config.json 已存在且本次配置未显式允许覆盖时抛出。
    ValueError
        配置加载或验证失败时抛出。
    """
    config_path = working_folder / "config.json"
    if config_dict is None:
        if config_path.exists():
            return load_config_json(config_path)
        effective_config = merge_config_dict()
        validate_config_dict(effective_config)
        return effective_config

    effective_config = merge_config_dict(config_dict)
    validate_config_dict(effective_config)
    if config_path.exists() and not _passed_config_allows_overwrite(config_dict):
        raise FileExistsError(
            f"配置文件已存在且传入配置未显式允许覆盖：{config_path}"
        )
    save_config_json_atomic(config_path, effective_config)
    return effective_config


def build_property_filter_mask(properties, filter_states):
    """根据动态布尔属性构建 AND 筛选 mask。

    本函数只读取 properties，不读取、复制或重建 waveform。返回值的
    第 i 个元素为 True，表示原始 epoch 下标 i 同时满足所有属性条件。

    Parameters
    ----------
    properties : Mapping
        属性名到一维布尔数组的映射，所有数组长度必须相同。
    filter_states : Mapping
        属性名到 ``all``、``true`` 或 ``false`` 的映射。未列出的
        属性按 ``all`` 处理。

    Returns
    -------
    numpy.ndarray
        一维 bool mask。返回的是新 mask，不会修改任何原属性数组。

    Raises
    ------
    ValueError
        输入不是 Mapping，属性不是等长一维布尔数据，或状态无效时抛出。

    Workflow
    --------
    1. 验证两个 Mapping，并拒绝 filter_states 中不存在的属性名。
    2. 将每个 property 验证为等长的一维 bool ndarray；只复制最终 mask。
    3. 从全 True mask 开始，对 true/false 条件逐项执行 NumPy AND，返回结果。
    """
    if not isinstance(properties, Mapping):
        raise ValueError("properties 必须是字典或其他 Mapping 对象。")
    if not isinstance(filter_states, Mapping):
        raise ValueError("filter_states 必须是字典或其他 Mapping 对象。")

    unknown_names = set(filter_states).difference(properties)
    if unknown_names:
        unknown_text = ", ".join(repr(name) for name in sorted(unknown_names, key=str))
        raise ValueError(f"filter_states 包含不存在的 property：{unknown_text}。")

    property_arrays = {}
    property_length = None
    for property_name, values in properties.items():
        if not isinstance(property_name, str) or not property_name:
            raise ValueError("property 名称必须是非空字符串。")
        array = np.asarray(values)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.bool_):
            raise ValueError(f"property {property_name!r} 必须是一维布尔数据。")
        if property_length is None:
            property_length = len(array)
        elif len(array) != property_length:
            raise ValueError(
                f"property {property_name!r} 长度为 {len(array)}，"
                f"期望长度为 {property_length}。"
            )
        property_arrays[property_name] = array

    if property_length is None:
        return np.empty(0, dtype=bool)

    mask = np.ones(property_length, dtype=bool)
    for property_name, array in property_arrays.items():
        state = filter_states.get(property_name, "all")
        if not isinstance(state, str) or state not in _VALID_FILTER_STATES:
            raise ValueError(
                f"属性 {property_name!r} 的筛选状态只能是 "
                f"'all'、'true' 或 'false'，当前值为 {state!r}。"
            )
        if state == "true":
            np.logical_and(mask, array, out=mask)
        elif state == "false":
            np.logical_and(mask, np.logical_not(array), out=mask)
    return mask


def _get_1d_source_metadata(source, source_name):
    """只通过元数据取得一维数据源长度和 dtype，不读取完整数据。

    Parameters
    ----------
    source : object
        支持 ``shape`` 或 ``len()`` 且支持下标/切片的一维数据源，可为 ndarray、
        memmap、h5py.Dataset 或自定义懒加载对象。
    source_name : str
        错误信息中的数据源名称。

    Returns
    -------
    tuple[int, numpy.dtype or None]
        第一项是一维长度，第二项是可解析 dtype；数据源无 dtype 时为 None。

    Workflow
    --------
    优先读取 shape 并验证恰为一维；否则使用 ndim/len；最后只解析 dtype 元数据。

    Raises
    ------
    ValueError
        数据源缺失、不可切片、不是一维、长度无效或 dtype 无法解析时抛出。
    """
    if source is None:
        raise ValueError(f"{source_name} 不能为 None。")
    if isinstance(source, (str, bytes)):
        raise ValueError(f"{source_name} 必须是一维数值数据，不能是字符串。")
    if not hasattr(source, "__getitem__"):
        raise ValueError(f"{source_name} 必须支持按下标或切片读取。")

    shape = getattr(source, "shape", None)
    if shape is not None:
        try:
            normalized_shape = tuple(shape)
        except TypeError as exc:
            raise ValueError(f"{source_name} 的 shape 无法解析。") from exc
        if len(normalized_shape) != 1:
            raise ValueError(
                f"{source_name} 必须是一维数据，当前 shape 为 {normalized_shape!r}。"
            )
        length = normalized_shape[0]
        if not _is_integer_scalar(length) or length < 0:
            raise ValueError(f"{source_name} 的长度无效：{length!r}。")
        length = int(length)
    else:
        ndim = getattr(source, "ndim", None)
        if ndim is not None and ndim != 1:
            raise ValueError(f"{source_name} 必须是一维数据，当前 ndim 为 {ndim!r}。")
        try:
            length = len(source)
        except (TypeError, OverflowError) as exc:
            raise ValueError(f"{source_name} 必须支持 len() 长度查询。") from exc

    dtype = getattr(source, "dtype", None)
    if dtype is not None:
        try:
            dtype = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError(f"{source_name} 的 dtype 无法解析：{dtype!r}。") from exc
    return length, dtype


def _validate_numeric_dtype(dtype, source_name, require_real):
    """使用 dtype 元数据验证数值类型。

    Parameters
    ----------
    dtype : numpy.dtype or None
        数据源 dtype；None 和 object 表示需要后续读取有限块检查内容。
    source_name : str
        错误信息中的数据源名称。
    require_real : bool
        True 时额外拒绝复数，适用于 timestamps。

    Returns
    -------
    bool
        dtype 已明确证明是合法数值类型时为 True；未知/object 时为 False。

    Workflow
    --------
    未知或 object 先返回 False；否则拒绝非数值、bool 以及不允许的复数类型。
    """
    if dtype is None or dtype.kind == "O":
        return False
    if not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_):
        raise ValueError(f"{source_name} 必须是数值数据，当前 dtype 为 {dtype}。")
    if require_real and np.issubdtype(dtype, np.complexfloating):
        raise ValueError(f"{source_name} 必须是实数时间戳，不能使用复数 dtype {dtype}。")
    return True


def _chunk_to_numeric_array(chunk, source_name, expected_length, require_real):
    """把一个有限数据块转换为一维数值 ndarray 并验证。

    Parameters
    ----------
    chunk : array-like
        已从大型数据源读取的小切片，不应是完整十亿级数据。
    source_name : str
        错误信息中的数据源名称。
    expected_length : int
        该切片按边界计算出的期望元素数。
    require_real : bool
        True 时拒绝复数元素。

    Returns
    -------
    numpy.ndarray
        形状为 ``(expected_length,)`` 的数值数组。

    Workflow
    --------
    转为 ndarray，验证维数和长度；object 数组逐元素检查，其余用 dtype 检查。

    Raises
    ------
    ValueError
        转换、形状、长度或数值类型不符合要求时抛出。
    """
    try:
        chunk_array = np.asarray(chunk)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_name} 的数据块无法转换为规则的一维数值数组。") from exc
    if chunk_array.ndim != 1:
        raise ValueError(
            f"{source_name} 必须是一维数据，读取到的数据块 shape 为 {chunk_array.shape!r}。"
        )
    if len(chunk_array) != expected_length:
        raise ValueError(
            f"{source_name} 的切片读取长度异常：期望 {expected_length}，"
            f"实际 {len(chunk_array)}。"
        )

    if chunk_array.dtype.kind == "O":
        for value in chunk_array:
            if _is_bool_scalar(value) or not isinstance(value, (Real, complex, np.number)):
                raise ValueError(f"{source_name} 必须只包含数值。")
            if require_real and isinstance(value, (complex, np.complexfloating)):
                raise ValueError(f"{source_name} 必须只包含实数时间戳。")
    else:
        _validate_numeric_dtype(chunk_array.dtype, source_name, require_real)
    return chunk_array


def _validate_1d_numeric_source(
        source,
        source_name,
        *,
        require_real=False,
        chunk_size=_DEFAULT_VALIDATION_CHUNK_SIZE,
):
    """验证通用一维数值数据源，已知 numeric dtype 时不读取其内容。

    Parameters
    ----------
    source : object
        ndarray、memmap、h5py.Dataset 或支持一维切片的懒数据源。
    source_name : str
        错误信息中的数据源名称。
    require_real : bool, keyword-only
        True 时只允许实数；False 时允许复数 waveform/value。
    chunk_size : int, keyword-only
        dtype 不明确时每次最多读取的元素数。

    Returns
    -------
    int
        数据源的一维元素数量。

    Workflow
    --------
    先读 shape/dtype；明确为数值时立即返回长度，否则按 chunk_size 分块检查内容。
    """
    length, dtype = _get_1d_source_metadata(source, source_name)
    dtype_is_numeric = _validate_numeric_dtype(dtype, source_name, require_real)
    if dtype_is_numeric or length == 0:
        return length

    _validate_positive_integer(chunk_size, "chunk_size")
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        try:
            chunk = source[start:end]
        except Exception as exc:
            raise ValueError(
                f"{source_name} 无法读取切片 [{start}:{end}]。"
            ) from exc
        _chunk_to_numeric_array(chunk, source_name, end - start, require_real)
    return length


def validate_timestamps_monotonic_chunked(
        timestamps_source,
        chunk_size=_DEFAULT_VALIDATION_CHUNK_SIZE,
        source_name="timestamps",
):
    """分块验证时间戳是一维实数且单调非递减。

    函数先通过 ``shape`` 或 ``len`` 获取总长度，然后每次只读取一个有限大小的
    切片。它既检查每个分块内部的顺序，也比较相邻分块的边界值。因此即使数据
    源有十亿个点，峰值内存仍主要由 ``chunk_size`` 决定，不会执行
    对完整数据源做全切片或把完整 Dataset 转换成 ndarray。

    Parameters
    ----------
    timestamps_source : object
        支持 ``len()``、一维切片读取，且包含真实数值时间戳的数据源。
    chunk_size : int, optional
        每次最多读取的时间戳数量，必须为正整数。
    source_name : str, optional
        错误信息中显示的数据源名称，便于定位具体通道和曲线。

    Returns
    -------
    None
        验证成功时不返回新对象。

    Raises
    ------
    ValueError
        数据不是一维实数、包含 NaN/Inf、切片行为异常或时间戳下降时抛出。

    Workflow
    --------
    1. 只通过 shape/len 取得总长度并验证 dtype 元数据。
    2. 每次读取至多 ``chunk_size`` 个点，检查有限性和块内单调性。
    3. 保存前一块末值，与下一块首值比较，从而覆盖分块边界。
    """
    _validate_positive_integer(chunk_size, "chunk_size")
    length, dtype = _get_1d_source_metadata(timestamps_source, source_name)
    _validate_numeric_dtype(dtype, source_name, require_real=True)

    previous_last = None
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        try:
            chunk = timestamps_source[start:end]
        except Exception as exc:
            raise ValueError(
                f"{source_name} 无法读取切片 [{start}:{end}]。"
            ) from exc
        chunk_array = _chunk_to_numeric_array(
            chunk,
            source_name,
            end - start,
            require_real=True,
        )
        if not np.all(np.isfinite(chunk_array)):
            raise ValueError(f"{source_name} 包含 NaN 或无穷大，不能作为有效时间戳。")
        if previous_last is not None and previous_last > chunk_array[0]:
            raise ValueError(f"{source_name} 在相邻分块边界处不是单调非递减。")
        if len(chunk_array) > 1 and np.any(chunk_array[1:] < chunk_array[:-1]):
            raise ValueError(f"{source_name} 不是单调非递减。")
        previous_last = chunk_array[-1]


def _validate_epochs_ids(epochs_ids, expected_length):
    """验证 epoch ID 的数量、字符串类型和唯一性。

    Parameters
    ----------
    epochs_ids : Sequence[str]
        一维 ID 容器，长度必须等于 epoch 数；每个元素必须是 Python str。
    expected_length : int
        根据 waveforms 得到的 epoch 数量。

    Returns
    -------
    None
        全部 ID 合法时不返回数据。

    Workflow
    --------
    排除单个字符串，读取容器长度，然后逐下标检查 str 类型并用 set 检查重复。

    Raises
    ------
    ValueError
        容器无长度、长度错误、元素不是字符串或存在重复 ID 时抛出。
    """
    if isinstance(epochs_ids, (str, bytes)):
        raise ValueError("epochs_ids 必须是一维字符串序列，不能是单个字符串。")
    try:
        actual_length = len(epochs_ids)
    except (TypeError, OverflowError) as exc:
        raise ValueError("epochs_ids 必须支持 len() 长度查询。") from exc
    if actual_length != expected_length:
        raise ValueError(
            f"epochs_ids 长度错误：期望 {expected_length}，实际 {actual_length}。"
        )

    seen_ids = set()
    for index in range(actual_length):
        epoch_id = epochs_ids[index]
        if not isinstance(epoch_id, str):
            raise ValueError(
                f"epochs_ids[{index}] 必须是字符串，当前类型为 "
                f"{type(epoch_id).__name__}。"
            )
        if epoch_id in seen_ids:
            raise ValueError(f"epochs_ids 必须唯一，发现重复 ID：{epoch_id!r}。")
        seen_ids.add(epoch_id)


def _validate_property_array(property_name, property_values, expected_length):
    """验证单个 property 为指定长度的一维布尔数据源。

    Parameters
    ----------
    property_name : str
        属性名称，用于生成准确错误信息。
    property_values : array-like of bool
        形状必须为 ``(expected_length,)``；可为 ndarray 或可切片懒数据源。
    expected_length : int
        epoch 总数。

    Returns
    -------
    None
        property 合法时不返回数据。

    Workflow
    --------
    先用元数据检查一维长度；dtype 明确为 bool 时结束，否则分块转换并验证 bool。
    """
    actual_length, dtype = _get_1d_source_metadata(
        property_values,
        f"property {property_name!r}",
    )
    if actual_length != expected_length:
        raise ValueError(
            f"property {property_name!r} 长度错误：期望 {expected_length}，"
            f"实际 {actual_length}。"
        )
    if dtype is not None and np.issubdtype(dtype, np.bool_):
        return

    for start in range(0, actual_length, _DEFAULT_VALIDATION_CHUNK_SIZE):
        end = min(start + _DEFAULT_VALIDATION_CHUNK_SIZE, actual_length)
        try:
            chunk = np.asarray(property_values[start:end])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"property {property_name!r} 无法读取为一维布尔数组。") from exc
        if chunk.ndim != 1:
            raise ValueError(f"property {property_name!r} 必须是一维布尔数组。")
        if not np.issubdtype(chunk.dtype, np.bool_):
            raise ValueError(
                f"property {property_name!r} 必须是布尔类型，当前 dtype 为 {chunk.dtype}。"
            )


def validate_epochs_dict(epochs_dict):
    """验证 epochs_dict，允许可选字段暂时缺失。

    验证过程逐个访问 waveform，不会把整个 ``waveforms`` 数据源转换成 ndarray。
    这样既能支持不等长 waveform，也为后续 HDF5 懒加载数据源保留接口。缺少
    ``epochs_ids``、``properties``、``is_delete`` 或 ``start_timestamps`` 本身
    不会报错；这些默认内容由 ``normalize_epochs_dict()`` 补齐。

    Parameters
    ----------
    epochs_dict : Mapping
        待验证的 epoch 数据字典。支持的完整键、list/ndarray 形状、各维度含义
        和缺失字段补齐规则，详见 ``DataViewer.__init__`` 的 ``epochs_dict`` 参数。

    Returns
    -------
    None
        验证成功时不返回新对象。

    Raises
    ------
    ValueError
        缺少 waveforms，waveform 不是一维数值数据，或任何可选字段格式错误时抛出。

    Workflow
    --------
    1. 验证顶层 Mapping 和必需的 waveforms 容器。
    2. 逐 epoch 验证每条 waveform 为一维数值源，允许不同长度且不合并它们。
    3. 分别检查可选 start_timestamps、epochs_ids 和每个动态 bool property。
    """
    if not isinstance(epochs_dict, Mapping):
        raise ValueError("epochs_dict 必须是字典或其他 Mapping 对象。")
    if "waveforms" not in epochs_dict or epochs_dict["waveforms"] is None:
        raise ValueError("epochs_dict 缺少必传字段 'waveforms'。")

    waveforms = epochs_dict["waveforms"]
    if isinstance(waveforms, (str, bytes)) or not hasattr(waveforms, "__getitem__"):
        raise ValueError("waveforms 必须支持 len() 和按 epoch 下标读取。")
    try:
        epoch_count = len(waveforms)
    except (TypeError, OverflowError) as exc:
        raise ValueError("waveforms 必须支持 len() 长度查询。") from exc

    for epoch_index in range(epoch_count):
        try:
            waveform = waveforms[epoch_index]
        except Exception as exc:
            raise ValueError(f"无法读取 waveform[{epoch_index}]。") from exc
        _validate_1d_numeric_source(
            waveform,
            f"waveform[{epoch_index}]",
            require_real=False,
        )

    start_timestamps = epochs_dict.get("start_timestamps")
    if start_timestamps is not None:
        timestamp_count = _validate_1d_numeric_source(
            start_timestamps,
            "start_timestamps",
            require_real=True,
        )
        if timestamp_count != epoch_count:
            raise ValueError(
                f"start_timestamps 长度错误：期望 {epoch_count}，实际 {timestamp_count}。"
            )

    epochs_ids = epochs_dict.get("epochs_ids")
    if epochs_ids is not None:
        _validate_epochs_ids(epochs_ids, epoch_count)

    properties = epochs_dict.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise ValueError("epochs_dict['properties'] 必须是字典。")
        for property_name, property_values in properties.items():
            _validate_property_array(property_name, property_values, epoch_count)


def normalize_epochs_dict(epochs_dict):
    """验证并补齐 epochs_dict 的可选字段。

    大型 waveform 数据源会原样保留引用，不会复制或整体转换。epoch ID 和
    properties 的规模通常只与 epoch 数量相同，因此标准化时将它们复制为 App
    可安全管理的内存对象，避免后续属性修改意外改变调用方传入的原始字典。

    Parameters
    ----------
    epochs_dict : Mapping
        原始 epoch 数据字典。输入容器结构及每个可选字段的缺省行为详见
        ``DataViewer.__init__`` 的 ``epochs_dict`` 参数。

    Returns
    -------
    dict
        包含 ``waveforms``、``start_timestamps``、``epochs_ids`` 和
        ``properties`` 的新字典。

    Raises
    ------
    ValueError
        输入数据不符合 ``validate_epochs_dict()`` 的规则时抛出。

    Workflow
    --------
    1. 先完整验证输入。
    2. 保留大型 waveforms 和 start_timestamps 的原引用，不复制其数据。
    3. 生成/复制 ID，把小型 properties 复制为 bool ndarray，并补齐 is_delete。
    """
    validate_epochs_dict(epochs_dict)

    normalized = dict(epochs_dict)
    waveforms = epochs_dict["waveforms"]
    epoch_count = len(waveforms)
    normalized["waveforms"] = waveforms
    normalized["start_timestamps"] = epochs_dict.get("start_timestamps")

    epochs_ids = epochs_dict.get("epochs_ids")
    if epochs_ids is None:
        normalized["epochs_ids"] = [f"epoch_{index}" for index in range(epoch_count)]
    else:
        normalized["epochs_ids"] = [epochs_ids[index] for index in range(epoch_count)]

    normalized_properties = {}
    properties = epochs_dict.get("properties")
    if properties is not None:
        for property_name, property_values in properties.items():
            normalized_properties[property_name] = np.asarray(
                property_values,
                dtype=bool,
            ).copy()
    if "is_delete" not in normalized_properties:
        normalized_properties["is_delete"] = np.zeros(epoch_count, dtype=bool)
    normalized["properties"] = normalized_properties
    return normalized


def validate_continuous_dict(continuous_dict):
    """验证动态通道、动态曲线组成的 continuous_dict。

    values 的长度和类型优先通过 ``shape``、``dtype`` 或 ``len`` 验证；具有数值
    dtype 的大数据源不会被读取。timestamps 必须检查内容顺序，因此使用
    ``validate_timestamps_monotonic_chunked()`` 分块扫描，并对共享公共时间轴只
    扫描一次。

    Parameters
    ----------
    continuous_dict : Mapping
        连续数据字典。``__common_timestamps__`` 是可选公共时间轴，其余键均被
        视为动态通道名称。支持的完整嵌套结构、每个数组维度的含义以及公共/
        独立时间轴缺省规则，详见 ``DataViewer.__init__`` 的 ``continuous_dict`` 参数。

    Returns
    -------
    None
        验证成功时不返回新对象。

    Raises
    ------
    ValueError
        通道/曲线结构错误、缺少 values、没有可用 timestamps、长度不同、类型
        错误或 timestamps 不是单调非递减时抛出。

    Workflow
    --------
    1. 如有公共时间轴，分块验证一次并记录其对象标识。
    2. 遍历动态通道和曲线，只用 shape/dtype/len 验证 values 元数据。
    3. 为每条曲线选择独立或公共 timestamps，检查等长，并对未检查过的时间轴
       执行分块单调性验证。
    """
    if not isinstance(continuous_dict, Mapping):
        raise ValueError("continuous_dict 必须是字典或其他 Mapping 对象。")

    common_timestamps = continuous_dict.get(_COMMON_TIMESTAMPS_KEY)
    common_length = None
    if common_timestamps is not None:
        common_length, _ = _get_1d_source_metadata(
            common_timestamps,
            "continuous_dict['__common_timestamps__']",
        )
        validate_timestamps_monotonic_chunked(
            common_timestamps,
            source_name="continuous_dict['__common_timestamps__']",
        )

    validated_timestamp_sources = set()
    if common_timestamps is not None:
        validated_timestamp_sources.add(id(common_timestamps))

    for channel_name, channel_curves in continuous_dict.items():
        if channel_name == _COMMON_TIMESTAMPS_KEY:
            continue
        if not isinstance(channel_curves, Mapping):
            raise ValueError(f"通道 {channel_name!r} 必须是包含曲线的字典。")

        for curve_name, curve_data in channel_curves.items():
            curve_path = f"通道 {channel_name!r} 的曲线 {curve_name!r}"
            if not isinstance(curve_data, Mapping):
                raise ValueError(f"{curve_path} 必须是字典。")
            if "values" not in curve_data or curve_data["values"] is None:
                raise ValueError(f"{curve_path} 缺少必传字段 'values'。")

            values = curve_data["values"]
            values_length, _ = _get_1d_source_metadata(values, f"{curve_path} values")

            own_timestamps = curve_data.get("timestamps")
            if own_timestamps is None:
                if common_timestamps is None:
                    raise ValueError(
                        f"{curve_path} 没有独立 timestamps，continuous_dict 也没有"
                        "可用的 __common_timestamps__。"
                    )
                timestamps = common_timestamps
                timestamps_length = common_length
            else:
                timestamps = own_timestamps
                timestamps_length, _ = _get_1d_source_metadata(
                    timestamps,
                    f"{curve_path} timestamps",
                )

            if timestamps_length != values_length:
                raise ValueError(
                    f"{curve_path} timestamps 长度为 {timestamps_length}，"
                    f"values 长度为 {values_length}，二者必须相同。"
                )

            _validate_1d_numeric_source(values, f"{curve_path} values")
            timestamps_identity = id(timestamps)
            if timestamps_identity not in validated_timestamp_sources:
                validate_timestamps_monotonic_chunked(
                    timestamps,
                    source_name=f"{curve_path} timestamps",
                )
                validated_timestamp_sources.add(timestamps_identity)


def normalize_continuous_dict(continuous_dict):
    """验证 continuous_dict，并为每条曲线填入实际使用的 timestamps 引用。

    本函数只新建字典容器，不复制 values 或 timestamps。缺少独立 timestamps 的
    曲线会直接引用公共时间轴，从而让后续代码无需重复判断缺失键，同时仍保持
    大型数据源的懒加载特性。

    Parameters
    ----------
    continuous_dict : Mapping
        原始连续数据字典。通道、曲线、values/timestamps 的形状和缺失字段处理
        详见 ``DataViewer.__init__`` 的 ``continuous_dict`` 参数。

    Returns
    -------
    dict
        结构独立、但大型数组仍共享原始引用的标准化字典。

    Raises
    ------
    ValueError
        输入数据不符合 ``validate_continuous_dict()`` 的规则时抛出。

    Workflow
    --------
    1. 完整验证动态 continuous 结构。
    2. 新建通道/曲线字典外壳，但保留大数组原引用。
    3. 曲线缺少自身 timestamps 时，将公共时间轴引用填入该曲线。
    """
    validate_continuous_dict(continuous_dict)

    normalized = {}
    common_timestamps = continuous_dict.get(_COMMON_TIMESTAMPS_KEY)
    if _COMMON_TIMESTAMPS_KEY in continuous_dict:
        normalized[_COMMON_TIMESTAMPS_KEY] = common_timestamps

    for channel_name, channel_curves in continuous_dict.items():
        if channel_name == _COMMON_TIMESTAMPS_KEY:
            continue
        normalized_curves = {}
        for curve_name, curve_data in channel_curves.items():
            normalized_curve = dict(curve_data)
            if normalized_curve.get("timestamps") is None:
                normalized_curve["timestamps"] = common_timestamps
            normalized_curves[curve_name] = normalized_curve
        normalized[channel_name] = normalized_curves
    return normalized


# ==================== HDF5 存储和懒加载算法 ====================

def _prepare_effective_config(config_dict):
    """合并并验证存储层使用的配置。

    Parameters
    ----------
    config_dict : Mapping[str, object] or None
        用户局部配置；None 表示全部使用默认值。详见 ``DataViewer.__init__`` 的
        ``config_dict`` 参数。

    Returns
    -------
    dict[str, object]
        深复制、字段完整且通过验证的配置。

    Workflow
    --------
    调用 ``merge_config_dict`` 补齐字段，再调用 ``validate_config_dict`` 检查。
    """
    effective_config = merge_config_dict(config_dict)
    validate_config_dict(effective_config)
    return effective_config


def _require_target_parent(file_path):
    """验证目标文件父目录存在且确实是目录。

    Parameters
    ----------
    file_path : str or os.PathLike[str]
        计划写入的 HDF5 文件路径。

    Returns
    -------
    pathlib.Path
        展开并解析后的目标绝对路径。

    Workflow
    --------
    解析目标路径，取得 parent，要求 parent 已存在且为目录。
    """
    path = Path(file_path).expanduser().resolve()
    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        raise ValueError(f"HDF5 目标目录不存在或不是目录：{parent}")
    return path


def _resolve_store_file_path(path_or_folder, default_filename):
    """把 Store 的目录或文件输入统一解析为 HDF5 文件路径。

    Parameters
    ----------
    path_or_folder : str or os.PathLike[str]
        已存在目录或具体文件路径。
    default_filename : str
        输入为目录时追加的固定文件名，例如 ``epochs.h5``。

    Returns
    -------
    pathlib.Path
        绝对文件路径；若输入是目录则为 ``目录/default_filename``。

    Workflow
    --------
    解析绝对路径；仅当它当前存在且是目录时追加文件名，否则视为文件路径。
    """
    path = Path(path_or_folder).expanduser().resolve()
    if path.exists() and path.is_dir():
        return path / default_filename
    return path


def _atomic_write_hdf5(file_path, writer, overwrite):
    """在目标目录写临时 HDF5，成功关闭后再原子替换目标文件。

    Parameters
    ----------
    file_path : str or Path
        最终 HDF5 文件路径。
    writer : callable
        接收临时文件路径的函数。该函数必须完整写入并关闭 HDF5 文件。
    overwrite : bool
        目标存在时是否允许替换。

    Returns
    -------
    Path
        已成功替换的最终绝对路径。

    Raises
    ------
    FileExistsError
        目标已经存在且不允许覆盖时抛出。
    ValueError
        目标父目录无效时抛出。

    Workflow
    --------
    1. 验证父目录和覆盖权限。
    2. 在相同目录创建临时路径，调用 ``writer(temp_path)`` 完整写入并关闭文件。
    3. 成功后原子替换目标；失败时删除临时文件并保留原目标。
    """
    target_path = _require_target_parent(file_path)
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"文件已存在且 overwrite=False：{target_path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        os.replace(temporary_path, target_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return target_path


def _infer_numeric_source_dtype(
        source,
        source_name,
        *,
        require_real=False,
        chunk_size=_DEFAULT_VALIDATION_CHUNK_SIZE,
):
    """分块推断一维数值数据源适合写入 HDF5 的 dtype。

    Parameters
    ----------
    source : object
        支持一维切片的 ndarray、Dataset 或懒数据源。
    source_name : str
        错误信息中的数据源名称。
    require_real : bool, keyword-only
        True 时不允许复数。
    chunk_size : int, keyword-only
        dtype 未明确时每次读取的最大元素数。

    Returns
    -------
    numpy.dtype
        可以容纳所有已检查数据块的统一 dtype；空源默认为 float64。

    Workflow
    --------
    明确的非 object dtype 直接验证并返回；否则逐块推断 dtype，再用
    ``numpy.result_type`` 合并为不丢失数值范围的统一类型。
    """
    length, dtype = _get_1d_source_metadata(source, source_name)
    if dtype is not None and dtype.kind != "O":
        _validate_numeric_dtype(dtype, source_name, require_real)
        return dtype

    inferred_dtype = None
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        chunk_array = _chunk_to_numeric_array(
            source[start:end],
            source_name,
            end - start,
            require_real,
        )
        if chunk_array.dtype.kind == "O":
            chunk_array = np.asarray(chunk_array.tolist())
            if chunk_array.dtype.kind == "O":
                raise ValueError(f"{source_name} 无法确定可存储的数值 dtype。")
            _validate_numeric_dtype(chunk_array.dtype, source_name, require_real)
        inferred_dtype = (
            chunk_array.dtype
            if inferred_dtype is None
            else np.result_type(inferred_dtype, chunk_array.dtype)
        )
    return np.dtype(np.float64 if inferred_dtype is None else inferred_dtype)


def _dataset_creation_kwargs(length, chunk_points, compression, compression_level):
    """生成一维可扩展、chunked HDF5 Dataset 的创建参数。

    Parameters
    ----------
    length : int
        Dataset 元素总数，可以为 0。
    chunk_points : int
        期望每个 chunk 的元素数；实际值不会超过数据长度且至少为 1。
    compression : str or None
        h5py 支持的压缩名称；None 表示不压缩。
    compression_level : int or None
        压缩级别；仅 compression 非 None 时写入参数。

    Returns
    -------
    dict[str, object]
        可直接展开给 ``create_dataset`` 的 shape、maxshape、chunks 和压缩参数。

    Workflow
    --------
    计算合法 chunk 长度，建立基本参数，再按需加入 compression 选项。
    """
    chunk_length = max(1, min(int(chunk_points), max(1, int(length))))
    kwargs = {
        "shape": (int(length),),
        "maxshape": (None,),
        "chunks": (chunk_length,),
    }
    if compression is not None:
        kwargs["compression"] = compression
        if compression_level is not None:
            kwargs["compression_opts"] = compression_level
    return kwargs


def _write_source_in_chunks(source, dataset, chunk_points, target_offset=0):
    """将一维数据源分块写入已创建的 HDF5 Dataset。

    Parameters
    ----------
    source : object
        支持 ``len`` 和一维切片的数据源，形状为 ``(N,)``。
    dataset : h5py.Dataset
        已预分配的目标一维 Dataset。
    chunk_points : int
        每次最多复制的元素数。
    target_offset : int, optional
        source[0] 在目标 Dataset 中的起始下标。

    Returns
    -------
    None
        写入完成后不返回数据。

    Workflow
    --------
    按 chunk_points 生成半开区间，读取有限切片、核对一维长度，再写入目标区间。
    """
    source_length = len(source)
    for start in range(0, source_length, chunk_points):
        end = min(start + chunk_points, source_length)
        chunk_array = np.asarray(source[start:end])
        if chunk_array.ndim != 1 or len(chunk_array) != end - start:
            raise ValueError(
                f"分块写入时读取到异常 shape {chunk_array.shape!r}，"
                f"期望长度为 {end - start}。"
            )
        dataset[target_offset + start:target_offset + end] = chunk_array


def _create_and_write_1d_dataset(
        parent_group,
        dataset_name,
        source,
        *,
        chunk_points,
        compression,
        compression_level,
        source_name,
        require_real=False,
):
    """推断 dtype、创建 chunked Dataset，并分块写入一维数据源。

    Parameters
    ----------
    parent_group : h5py.Group or h5py.File
        新 Dataset 的父对象。
    dataset_name : str
        Dataset 名称，不包含父路径。
    source : object
        形状为 ``(N,)`` 的数值数据源。
    chunk_points : int, keyword-only
        每个 HDF5 chunk 及每次复制的目标点数。
    compression : str or None, keyword-only
        HDF5 压缩算法。
    compression_level : int or None, keyword-only
        压缩级别。
    source_name : str, keyword-only
        错误信息中的数据源名称。
    require_real : bool, keyword-only
        True 时拒绝复数。

    Returns
    -------
    h5py.Dataset
        已创建并写完的一维 Dataset 对象。

    Workflow
    --------
    读取长度元数据、分块推断 dtype、生成 Dataset 参数、创建对象并分块复制。
    """
    length, _ = _get_1d_source_metadata(source, source_name)
    dtype = _infer_numeric_source_dtype(
        source,
        source_name,
        require_real=require_real,
        chunk_size=chunk_points,
    )
    kwargs = _dataset_creation_kwargs(
        length,
        chunk_points,
        compression,
        compression_level,
    )
    dataset = parent_group.create_dataset(dataset_name, dtype=dtype, **kwargs)
    _write_source_in_chunks(source, dataset, chunk_points)
    return dataset


def _decode_utf8(value):
    """把 HDF5 attribute 值统一还原为 Python 字符串。

    Parameters
    ----------
    value : bytes or object
        h5py 返回的 attribute 值。

    Returns
    -------
    str
        bytes 按 UTF-8 解码；其他对象使用 ``str`` 转换。

    Workflow
    --------
    判断 bytes 类型并解码，否则直接生成字符串表示。
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _set_utf8_attribute(hdf5_object, key, value):
    """以 UTF-8 variable-length string 保存 HDF5 attribute。

    Parameters
    ----------
    hdf5_object : h5py.File, h5py.Group or h5py.Dataset
        接收 attribute 的 HDF5 对象。
    key : str
        attribute 名称。
    value : object
        会先转换为 str 的 attribute 值。

    Returns
    -------
    None
        attribute 创建完成后不返回数据。

    Workflow
    --------
    创建 UTF-8 可变长字符串 dtype，再通过 attrs.create 写入字符串。
    """
    hdf5_object.attrs.create(
        key,
        str(value),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _normalize_indices(indices, upper_bound, field_name="indices"):
    """将单个或多个下标转换为 Python int 列表并检查范围。

    Parameters
    ----------
    indices : Integral or Iterable[Integral]
        单个下标或可迭代下标；允许重复，保持输入顺序。
    upper_bound : int
        合法半开区间的上界，即 ``0 <= index < upper_bound``。
    field_name : str, optional
        错误信息中显示的字段名称。

    Returns
    -------
    list[int]
        已转换且范围合法的 Python 整数列表。

    Workflow
    --------
    统一包装成列表，逐项排除 bool/非整数并检查上下界，然后追加 int 值。
    """
    if _is_integer_scalar(indices):
        raw_indices = [int(indices)]
    else:
        try:
            raw_indices = list(indices)
        except TypeError as exc:
            raise ValueError(f"{field_name} 必须是整数或整数序列。") from exc

    normalized = []
    for value in raw_indices:
        if not _is_integer_scalar(value):
            raise ValueError(f"{field_name} 必须只包含整数，发现 {value!r}。")
        index = int(value)
        if index < 0 or index >= upper_bound:
            raise IndexError(
                f"{field_name} 中的下标 {index} 超出范围 [0, {upper_bound})。"
            )
        normalized.append(index)
    return normalized


def _iter_contiguous_index_ranges(sorted_indices):
    """把已排序唯一整数下标合并为半开连续区间。

    Parameters
    ----------
    sorted_indices : Sequence[int]
        严格升序且已去重的整数下标，例如 ``[1, 2, 5]``。

    Yields
    ------
    tuple[int, int]
        可直接用于切片的 ``(start, end)``，示例输入产生 ``(1, 3)``、``(5, 6)``。

    Workflow
    --------
    从第一个下标开始扫描；遇到不连续值时产出上一段，结束后产出最后一段。
    """
    if not sorted_indices:
        return
    range_start = sorted_indices[0]
    previous = range_start
    for index in sorted_indices[1:]:
        if index != previous + 1:
            yield range_start, previous + 1
            range_start = index
        previous = index
    yield range_start, previous + 1


def find_timestamp_slice(timestamps_source, start_timestamp, end_timestamp):
    """查找闭区间时间窗口对应的半开数组切片。

    返回的 ``start_index`` 指向第一个大于等于 ``start_timestamp`` 的时间戳，
    ``end_index`` 指向第一个严格大于 ``end_timestamp`` 的时间戳。因此读取
    ``timestamps[start_index:end_index]`` 会包含恰好等于结束时间的点。

    小型 NumPy 数组直接使用 ``np.searchsorted``。HDF5 Dataset 和其他懒数据源
    使用标量二分查找，每次只读取一个时间戳，时间复杂度为 ``O(log N)``，不会
    对 Dataset 做全切片或把完整时间轴转换成 ndarray。

    Parameters
    ----------
    timestamps_source : object
        单调非递减的一维时间戳数据源。
    start_timestamp : Real
        窗口开始时间。
    end_timestamp : Real
        窗口结束时间。

    Returns
    -------
    tuple[int, int]
        可直接用于半开切片的开始和结束下标。

    Raises
    ------
    ValueError
        时间边界不是有限实数、开始时间大于结束时间或数据源不是一维时抛出。

    Workflow
    --------
    1. 验证闭区间的两个有限边界及一维数据源长度。
    2. 普通 ndarray 使用 ``searchsorted``；其他懒数据源执行两次标量二分查找。
    3. 左边界使用 left 语义、右边界使用 right 语义，返回可含等于结束值的半开切片。
    """
    if not _is_real_scalar(start_timestamp) or not _is_real_scalar(end_timestamp):
        raise ValueError("start_timestamp 和 end_timestamp 必须是有限实数。")
    if start_timestamp > end_timestamp:
        raise ValueError("start_timestamp 不能大于 end_timestamp。")

    length, _ = _get_1d_source_metadata(timestamps_source, "timestamps_source")
    if isinstance(timestamps_source, np.ndarray):
        start_index = int(np.searchsorted(timestamps_source, start_timestamp, side="left"))
        end_index = int(np.searchsorted(timestamps_source, end_timestamp, side="right"))
        return start_index, end_index

    def lazy_searchsorted(target, side):
        """在懒时间轴上执行一次 searchsorted 语义的标量二分。

        Parameters
        ----------
        target : Real
            需要定位的时间值。
        side : {"left", "right"}
            left 返回首个大于等于 target 的位置；right 返回首个大于 target 的位置。

        Returns
        -------
        int
            范围为 ``[0, length]`` 的插入下标。

        Workflow
        --------
        维护左闭右开搜索范围，每轮只读取中点一个标量，并按 side 规则收缩范围。
        """
        left = 0
        right = length
        while left < right:
            middle = (left + right) // 2
            middle_value = timestamps_source[middle]
            move_left = middle_value < target if side == "left" else middle_value <= target
            if move_left:
                left = middle + 1
            else:
                right = middle
        return left

    return (
        lazy_searchsorted(start_timestamp, "left"),
        lazy_searchsorted(end_timestamp, "right"),
    )


def downsample_min_max(timestamps, values, target_width, max_points_per_pixel):
    """使用 min-max envelope 将当前时间窗口压缩到可绘制点数。

    算法把输入按连续位置分箱，每个分箱保留最小值、最大值及其原始时间位置，
    并按原下标顺序输出。因此尖峰不会像简单 ``values[::step]`` 那样被跳过。
    当输入点数已经不超过 Canvas 容量时，直接返回原对象，不进行复制。

    Parameters
    ----------
    timestamps : array-like
        当前窗口的一维时间戳。
    values : array-like
        与 timestamps 等长的一维数值。
    target_width : Real
        Canvas 的目标像素宽度。
    max_points_per_pixel : Real
        每个像素允许交给 Matplotlib 的最大点数。

    Returns
    -------
    tuple[array-like, array-like]
        下采样后的时间戳和值；无需下采样时就是原输入对象。

    Raises
    ------
    ValueError
        输入不是等长一维数据，或目标参数不是正数时抛出。

    Workflow
    --------
    1. 验证像素参数，把当前有限时间窗切片转换为等长一维 ndarray。
    2. 数据量未超过像素容量时直接返回原对象，避免无意义复制。
    3. 超出时按连续下标分箱，每箱保留最小/最大值的原下标，并按时间顺序输出。
    """
    _validate_positive_number(target_width, "target_width")
    _validate_positive_number(max_points_per_pixel, "max_points_per_pixel")

    timestamps_array = np.asarray(timestamps)
    values_array = np.asarray(values)
    if timestamps_array.ndim != 1 or values_array.ndim != 1:
        raise ValueError("timestamps 和 values 必须都是一维数组。")
    if len(timestamps_array) != len(values_array):
        raise ValueError(
            f"timestamps 长度为 {len(timestamps_array)}，values 长度为 "
            f"{len(values_array)}，二者必须相同。"
        )

    maximum_output_points = max(2, int(target_width * max_points_per_pixel))
    point_count = len(values_array)
    if point_count <= maximum_output_points:
        return timestamps, values

    bin_count = max(1, maximum_output_points // 2)
    bin_edges = np.linspace(0, point_count, bin_count + 1, dtype=np.int64)
    selected_indices = np.empty(bin_count * 2, dtype=np.int64)
    selected_count = 0

    for bin_index in range(bin_count):
        start = int(bin_edges[bin_index])
        end = int(bin_edges[bin_index + 1])
        bin_values = values_array[start:end]
        minimum_index = start + int(np.argmin(bin_values))
        maximum_index = start + int(np.argmax(bin_values))
        if minimum_index == maximum_index:
            selected_indices[selected_count] = minimum_index
            selected_count += 1
        elif minimum_index < maximum_index:
            selected_indices[selected_count:selected_count + 2] = (
                minimum_index,
                maximum_index,
            )
            selected_count += 2
        else:
            selected_indices[selected_count:selected_count + 2] = (
                maximum_index,
                minimum_index,
            )
            selected_count += 2

    selected_indices = selected_indices[:selected_count]
    return timestamps_array[selected_indices], values_array[selected_indices]


class EpochsHDF5Store:
    """管理不等长 epoch waveform 和布尔 properties 的 HDF5 文件句柄。

    大型 waveform 保存在一维 values Dataset 中，通过 int64 offsets 定位；只有
    offsets、ID 和形状为 ``(epoch_count,)`` 的布尔 properties 常驻内存。
    """

    def __init__(self, file_path, mode="r+"):
        """打开已有 epochs.h5 并建立轻量级内存索引。

        Parameters
        ----------
        file_path : str or Path
            已存在的 epochs HDF5 文件。
        mode : {"r", "r+"}, optional
            ``r`` 只读；``r+`` 允许局部写 properties。

        Returns
        -------
        None
            构造函数不返回值；成功后实例持有已打开文件和轻量级索引。

        Raises
        ------
        ValueError
            文件 schema 不受支持或结构损坏时抛出。

        Workflow
        --------
        1. 解析目录/文件输入并用 h5py 打开句柄。
        2. 校验 schema 和必需 Dataset，同时建立轻量级内存索引。
        3. 校验失败时立即关闭文件，再把原异常交给调用方。
        """
        if mode not in {"r", "r+"}:
            raise ValueError("EpochsHDF5Store mode 只能是 'r' 或 'r+'。")
        self.file_path = _resolve_store_file_path(file_path, "epochs.h5")
        self.mode = mode
        self._file = h5py.File(self.file_path, mode)
        try:
            self._load_and_validate_schema()
        except Exception:
            self._file.close()
            raise

    @classmethod
    def create(
            cls,
            file_path,
            epochs_dict,
            config_dict=None,
            *,
            overwrite=False,
    ):
        """验证数据并原子创建 epochs HDF5，成功后以可写模式返回 Store。

        waveforms 先逐条计算长度和 dtype，再预创建扁平 values Dataset。写入时
        每次只复制一个有限块，绝不使用 ``np.concatenate()``。

        Parameters
        ----------
        file_path : str or os.PathLike[str]
            最终 epochs.h5 路径，或已经存在的 working folder。
        epochs_dict : Mapping[str, object]
            必须含 ``waveforms``。waveforms 是长度 E 的容器，每个元素形状为
            ``(Ni,)``；可选 start_timestamps/epochs_ids/properties 长度均为 E。
            完整支持形式、维度语义和缺省处理详见 ``DataViewer.__init__`` 的
            ``epochs_dict`` 参数。
        config_dict : Mapping[str, object] or None, optional
            局部或完整配置；storage 区块控制 chunk 和压缩。默认值来源详见
            ``DataViewer.__init__`` 的 ``config_dict`` 参数。
        overwrite : bool, keyword-only
            目标存在时是否允许原子替换。

        Returns
        -------
        EpochsHDF5Store
            指向最终文件、mode 为 ``"r+"`` 的已打开 Store；调用方负责关闭。

        Workflow
        --------
        1. 合并配置并标准化 epoch 数据。
        2. 逐 waveform 计算长度、offsets 和统一 dtype，不合并全部数据。
        3. 在临时 HDF5 中预创建 Dataset，逐条分块写入，再原子替换并重新打开。

        Raises
        ------
        ValueError
            数据结构、属性名称或配置无效时抛出。
        FileExistsError
            目标存在且 overwrite=False 时抛出。
        """
        effective_config = _prepare_effective_config(config_dict)
        normalized = normalize_epochs_dict(epochs_dict)
        storage_config = effective_config["storage"]
        chunk_points = storage_config["waveforms_chunk_points"]

        waveforms = normalized["waveforms"]
        epoch_count = len(waveforms)
        offsets = np.empty(epoch_count + 1, dtype=np.int64)
        offsets[0] = 0
        waveform_dtypes = []
        total_points = 0
        for epoch_index in range(epoch_count):
            waveform = waveforms[epoch_index]
            waveform_length, _ = _get_1d_source_metadata(
                waveform,
                f"waveform[{epoch_index}]",
            )
            total_points += waveform_length
            if total_points > np.iinfo(np.int64).max:
                raise ValueError("全部 waveform 总点数超过 int64 offsets 可表示范围。")
            offsets[epoch_index + 1] = total_points
            if waveform_length:
                waveform_dtypes.append(
                    _infer_numeric_source_dtype(
                        waveform,
                        f"waveform[{epoch_index}]",
                        chunk_size=chunk_points,
                    )
                )
        values_dtype = (
            np.dtype(np.float64)
            if not waveform_dtypes
            else np.result_type(*waveform_dtypes)
        )

        def writer(temporary_path):
            """把完整 epoch schema 写入一个临时路径并在返回前关闭句柄。

            Parameters
            ----------
            temporary_path : pathlib.Path
                `_atomic_write_hdf5` 在目标同目录创建的临时文件路径。

            Returns
            -------
            None
                所有 Dataset 写入并 flush 后返回。

            Workflow
            --------
            建立 metadata 和各 Group，预分配扁平 values，逐 waveform 分块复制，
            再写入可选时间戳、UTF-8 ID 和动态 bool properties。
            """
            with h5py.File(temporary_path, "w") as hdf5_file:
                hdf5_file.attrs["schema_version"] = _epochs_SCHEMA_VERSION
                _set_utf8_attribute(hdf5_file, "store_type", "epochs")
                hdf5_file.attrs["has_start_timestamps"] = (
                    normalized["start_timestamps"] is not None
                )

                waveforms_group = hdf5_file.create_group("waveforms")
                values_kwargs = _dataset_creation_kwargs(
                    total_points,
                    chunk_points,
                    storage_config["waveforms_compression"],
                    storage_config["waveforms_compression_level"],
                )
                values_dataset = waveforms_group.create_dataset(
                    "values",
                    dtype=values_dtype,
                    **values_kwargs,
                )
                waveforms_group.create_dataset("offsets", data=offsets, dtype=np.int64)

                for epoch_index in range(epoch_count):
                    _write_source_in_chunks(
                        waveforms[epoch_index],
                        values_dataset,
                        chunk_points,
                        target_offset=int(offsets[epoch_index]),
                    )

                start_timestamps = normalized["start_timestamps"]
                if start_timestamps is not None:
                    _create_and_write_1d_dataset(
                        hdf5_file,
                        "start_timestamps",
                        start_timestamps,
                        chunk_points=chunk_points,
                        compression=None,
                        compression_level=None,
                        source_name="start_timestamps",
                        require_real=True,
                    )

                id_dtype = h5py.string_dtype(encoding="utf-8")
                ids_dataset = hdf5_file.create_dataset(
                    "epochs_ids",
                    shape=(epoch_count,),
                    dtype=id_dtype,
                )
                if epoch_count:
                    ids_dataset[...] = np.asarray(normalized["epochs_ids"], dtype=object)

                properties_group = hdf5_file.create_group("properties")
                property_chunk_size = storage_config["properties_chunk_size"]
                for property_name, property_values in normalized["properties"].items():
                    if not isinstance(property_name, str) or not property_name:
                        raise ValueError("property 名称必须是非空字符串。")
                    if "/" in property_name:
                        raise ValueError(
                            f"property 名称 {property_name!r} 不能包含 HDF5 路径分隔符 '/'."
                        )
                    property_kwargs = _dataset_creation_kwargs(
                        epoch_count,
                        property_chunk_size,
                        None,
                        None,
                    )
                    property_dataset = properties_group.create_dataset(
                        property_name,
                        dtype=np.bool_,
                        **property_kwargs,
                    )
                    if epoch_count:
                        property_dataset[...] = property_values
                hdf5_file.flush()

        target_path = _atomic_write_hdf5(
            _resolve_store_file_path(file_path, "epochs.h5"),
            writer,
            overwrite,
        )
        return cls(target_path, mode="r+")

    def _load_and_validate_schema(self):
        """验证 epoch 文件结构并建立轻量级内存索引。

        Parameters
        ----------
        self : EpochsHDF5Store
            `_file` 已打开的 Store 实例。

        Returns
        -------
        None
            成功后设置 values_dataset、offsets、epoch_count、epochs_ids、
            start_timestamps 和 properties 等实例属性。

        Workflow
        --------
        检查根 metadata 和必需路径；验证 values/offsets 对应关系；加载小型 ID、
        offsets、properties；start_timestamps 只保留 Dataset 引用。

        Raises
        ------
        ValueError
            schema 版本、路径、dtype、长度或 metadata 不一致时抛出。
        """
        schema_version = self._file.attrs.get("schema_version")
        store_type = _decode_utf8(self._file.attrs.get("store_type", ""))
        if schema_version != _epochs_SCHEMA_VERSION or store_type != "epochs":
            raise ValueError(
                f"不支持的 epochs HDF5 schema：version={schema_version!r}, "
                f"store_type={store_type!r}。"
            )
        required_paths = (
            "waveforms/values",
            "waveforms/offsets",
            "epochs_ids",
            "properties",
        )
        for required_path in required_paths:
            if required_path not in self._file:
                raise ValueError(f"epochs HDF5 缺少必需路径 /{required_path}。")

        self.values_dataset = self._file["waveforms/values"]
        if self.values_dataset.ndim != 1:
            raise ValueError("/waveforms/values 必须是一维 Dataset。")
        if self.values_dataset.chunks is None:
            raise ValueError("/waveforms/values 必须使用 chunked Dataset。")
        _validate_numeric_dtype(
            self.values_dataset.dtype,
            "/waveforms/values",
            require_real=False,
        )
        offsets_dataset = self._file["waveforms/offsets"]
        if offsets_dataset.ndim != 1 or offsets_dataset.dtype != np.dtype(np.int64):
            raise ValueError("/waveforms/offsets 必须是一维 int64 Dataset。")
        self.offsets = offsets_dataset[()]
        if len(self.offsets) == 0 or self.offsets[0] != 0:
            raise ValueError("/waveforms/offsets 必须至少包含起始值 0。")
        if np.any(self.offsets[1:] < self.offsets[:-1]):
            raise ValueError("/waveforms/offsets 必须单调非递减。")
        if int(self.offsets[-1]) != len(self.values_dataset):
            raise ValueError("offsets 最后一个值必须等于 /waveforms/values 长度。")

        self.epoch_count = len(self.offsets) - 1
        ids_dataset = self._file["epochs_ids"]
        if ids_dataset.ndim != 1 or len(ids_dataset) != self.epoch_count:
            raise ValueError("/epochs_ids 长度必须等于 epoch 数量。")
        self.epochs_ids = ids_dataset.asstr()[()].tolist()
        if len(set(self.epochs_ids)) != self.epoch_count:
            raise ValueError("/epochs_ids 中存在重复 ID。")

        has_start_timestamps = bool(
            self._file.attrs.get("has_start_timestamps", "start_timestamps" in self._file)
        )
        if has_start_timestamps != ("start_timestamps" in self._file):
            raise ValueError("start_timestamps metadata 与实际 Dataset 不一致。")
        self.start_timestamps = self._file.get("start_timestamps")
        if self.start_timestamps is not None:
            if self.start_timestamps.ndim != 1:
                raise ValueError("/start_timestamps 必须是一维 Dataset。")
            if len(self.start_timestamps) != self.epoch_count:
                raise ValueError("/start_timestamps 长度必须等于 epoch 数量。")
            _validate_numeric_dtype(
                self.start_timestamps.dtype,
                "/start_timestamps",
                require_real=True,
            )

        self.properties = {}
        for property_name, dataset in self._file["properties"].items():
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"/properties/{property_name} 必须是 Dataset。")
            if dataset.ndim != 1 or len(dataset) != self.epoch_count:
                raise ValueError(
                    f"property {property_name!r} 长度必须等于 {self.epoch_count}。"
                )
            if not np.issubdtype(dataset.dtype, np.bool_):
                raise ValueError(f"property {property_name!r} 必须是布尔 Dataset。")
            self.properties[property_name] = dataset[()].astype(bool, copy=False)
        if "is_delete" not in self.properties:
            raise ValueError("epochs HDF5 缺少 /properties/is_delete。")

    def _ensure_open(self):
        """确认当前 HDF5 句柄仍然有效。

        Returns
        -------
        None
            句柄有效时不返回数据。

        Workflow
        --------
        检查 `_file` 是否存在及 h5py id.valid；关闭状态抛出 RuntimeError。
        """
        if self._file is None or not self._file.id.valid:
            raise RuntimeError("EpochsHDF5Store 已关闭。")

    def __len__(self):
        """返回 epoch 数量。

        Returns
        -------
        int
            ``len(offsets) - 1``，不读取 waveform Dataset。

        Workflow
        --------
        直接返回 schema 加载时缓存的 ``epoch_count``。
        """
        return self.epoch_count

    def get_waveform(self, epoch_index):
        """懒加载一个 epoch 对应的 waveform。

        Parameters
        ----------
        epoch_index : Integral
            原始 epoch 下标，合法范围为 ``[0, epoch_count)``。

        Returns
        -------
        numpy.ndarray
            形状为 ``(Ni,)`` 的单条 waveform；Ni 可与其他 epoch 不同。

        Workflow
        --------
        验证句柄和下标，从 offsets 读取起止位置，仅切片 values[start:end]。
        """
        self._ensure_open()
        indices = _normalize_indices(epoch_index, self.epoch_count, "epoch_index")
        index = indices[0]
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.values_dataset[start:end]

    def get_properties(self, *, copy=True):
        """返回全部动态布尔 properties 的内存镜像。

        Parameters
        ----------
        copy : bool, keyword-only
            True 返回每个数组的副本；False 返回 Store 管理的可写原数组。

        Returns
        -------
        dict[str, numpy.ndarray]
            属性名到形状 ``(epoch_count,)``、dtype bool 数组的映射。

        Workflow
        --------
        先确认句柄有效，再按 copy 选择逐数组复制或返回原字典。
        """
        self._ensure_open()
        if copy:
            return {name: values.copy() for name, values in self.properties.items()}
        return self.properties

    def get_property(self, property_name, *, copy=True):
        """读取一个 property 的内存数组。

        Parameters
        ----------
        property_name : str
            HDF5 `/properties` 下的真实属性名。
        copy : bool, keyword-only
            True 返回副本；False 返回可直接修改的内存镜像。

        Returns
        -------
        numpy.ndarray
            形状 ``(epoch_count,)``、dtype bool 的数组。

        Workflow
        --------
        验证句柄和属性名，再按 copy 选择副本或原数组。
        """
        self._ensure_open()
        if property_name not in self.properties:
            raise KeyError(f"不存在 property {property_name!r}。")
        values = self.properties[property_name]
        return values.copy() if copy else values

    def write_property_indices(self, property_name, indices, values=None):
        """只把指定下标的布尔属性按连续区间写回 HDF5。

        ``values=None`` 时从 ``get_property(copy=False)`` 返回的内存数组写回；传入
        values 时会先更新内存数组。重复下标会去重，传值时最后一次赋值生效。

        Returns
        -------
        int
            实际写入的唯一 epoch 下标数量。

        Parameters
        ----------
        property_name : str
            需要局部写回的 bool property 名称。
        indices : Integral or Iterable[Integral]
            一个或多个原始 epoch 下标；允许无序和重复。
        values : bool, Iterable[bool] or None, optional
            None 表示从当前内存镜像取值；标量会广播；序列必须与原 indices 等长。

        Workflow
        --------
        1. 验证可写模式、属性名和下标。
        2. 如传 values，按“重复下标最后一次赋值生效”更新内存镜像。
        3. 排序去重、合并连续区间，仅写对应 Dataset 切片并返回唯一数量。

        Raises
        ------
        PermissionError
            Store 为只读模式时抛出。
        KeyError, ValueError or IndexError
            属性、值或下标无效时抛出。
        """
        self._ensure_open()
        if self.mode == "r":
            raise PermissionError("Store 以只读模式打开，不能写入 property。")
        if property_name not in self.properties:
            raise KeyError(f"不存在 property {property_name!r}。")

        raw_indices = _normalize_indices(indices, self.epoch_count)
        if not raw_indices:
            return 0

        property_array = self.properties[property_name]
        if values is not None:
            if _is_bool_scalar(values):
                raw_values = [bool(values)] * len(raw_indices)
            else:
                try:
                    raw_values = list(values)
                except TypeError as exc:
                    raise ValueError("property values 必须是布尔值或布尔序列。") from exc
                if len(raw_values) != len(raw_indices):
                    raise ValueError(
                        f"property values 长度为 {len(raw_values)}，indices 长度为 "
                        f"{len(raw_indices)}，二者必须相同。"
                    )
                if not all(_is_bool_scalar(value) for value in raw_values):
                    raise ValueError("property values 必须只包含布尔值。")
                raw_values = [bool(value) for value in raw_values]

            latest_values = {}
            for index, value in zip(raw_indices, raw_values):
                latest_values[index] = value
            for index, value in latest_values.items():
                property_array[index] = value
            unique_indices = sorted(latest_values)
        else:
            unique_indices = sorted(set(raw_indices))

        dataset = self._file[f"properties/{property_name}"]
        for start, end in _iter_contiguous_index_ranges(unique_indices):
            dataset[start:end] = property_array[start:end]
        return len(unique_indices)

    def flush(self):
        """把 HDF5 缓冲区刷新到磁盘。

        Returns
        -------
        None
            h5py flush 完成后返回。

        Workflow
        --------
        先确认句柄有效，再调用 ``h5py.File.flush``。
        """
        self._ensure_open()
        self._file.flush()

    def close(self):
        """幂等关闭 HDF5 文件句柄。

        Returns
        -------
        None
            已关闭或关闭完成均不返回数据。

        Workflow
        --------
        仅当 `_file` 存在且 id.valid 时调用 close，因此重复调用安全。
        """
        if self._file is not None and self._file.id.valid:
            self._file.close()

    def __enter__(self):
        """进入 with 上下文并返回已打开的 Store。

        Returns
        -------
        EpochsHDF5Store
            当前实例。

        Workflow
        --------
        验证句柄仍有效，然后返回 self。
        """
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出 with 上下文并关闭 Store，不吞掉异常。

        Parameters
        ----------
        exc_type : type[BaseException] or None
            上下文异常类型。
        exc_value : BaseException or None
            异常实例。
        traceback : types.TracebackType or None
            异常回溯对象。

        Returns
        -------
        bool
            固定 False，表示原异常应继续向外传播。

        Workflow
        --------
        幂等调用 close，再返回 False。
        """
        self.close()
        return False



class ContinuousHDF5Store:
    """管理动态通道、动态曲线 continuous HDF5 的懒加载访问。

    实例只在内存中保存通道/曲线名称到 HDF5 Group 路径的索引；values 与
    timestamps 始终保留为 Dataset，仅在给定时间窗内切片读取。
    """

    def __init__(self, file_path, mode="r"):
        """打开已有 continuous.h5 并建立名称到内部 Group 的小型索引。

        Parameters
        ----------
        file_path : str or os.PathLike[str]
            已有 continuous.h5 路径，或包含该固定文件名的工作目录。
        mode : {"r", "r+"}, optional
            HDF5 打开模式；Detail 通常使用只读 ``"r"``。

        Returns
        -------
        None
            构造函数不返回值；成功后实例持有打开的 h5py.File。

        Workflow
        --------
        解析路径、打开句柄、验证 schema 并建立名称索引；任何失败都会先关闭句柄。
        """
        if mode not in {"r", "r+"}:
            raise ValueError("ContinuousHDF5Store mode 只能是 'r' 或 'r+'。")
        self.file_path = _resolve_store_file_path(file_path, "continuous.h5")
        self.mode = mode
        self._file = h5py.File(self.file_path, mode)
        try:
            self._load_and_validate_schema()
        except Exception:
            self._file.close()
            raise

    @classmethod
    def create(
            cls,
            file_path,
            continuous_dict,
            config_dict=None,
            *,
            overwrite=False,
    ):
        """验证并原子创建 continuous HDF5，所有大数组均分块复制。

        Parameters
        ----------
        file_path : str or os.PathLike[str]
            最终 continuous.h5 路径或 working folder。
        continuous_dict : Mapping[str, object]
            动态通道字典；保留键 ``__common_timestamps__`` 可保存形状 ``(N,)``
            的公共时间轴。每条曲线含形状 ``(M,)`` 的 values 和可选 timestamps。
            完整支持形式、维度语义和缺省处理详见 ``DataViewer.__init__`` 的
            ``continuous_dict`` 参数。
        config_dict : Mapping[str, object] or None, optional
            控制 continuous chunk、压缩和验证的配置。默认值来源详见
            ``DataViewer.__init__`` 的 ``config_dict`` 参数。
        overwrite : bool, keyword-only
            目标已存在时是否允许原子替换。

        Returns
        -------
        ContinuousHDF5Store
            指向最终文件、只读模式的已打开 Store。

        Workflow
        --------
        1. 合并配置并验证所有曲线和时间轴。
        2. 临时文件中公共时间轴只写一次，通道/曲线使用编号 Group 和 UTF-8 名称。
        3. 所有一维数据分块写入，flush 后原子替换，再以只读模式打开。
        """
        effective_config = _prepare_effective_config(config_dict)
        validate_continuous_dict(continuous_dict)
        storage_config = effective_config["storage"]
        chunk_points = storage_config["continuous_chunk_points"]
        compression = storage_config["continuous_compression"]
        compression_level = storage_config["continuous_compression_level"]
        common_timestamps = continuous_dict.get(_COMMON_TIMESTAMPS_KEY)

        def writer(temporary_path):
            """把 continuous schema 写入临时文件并完整关闭。

            Parameters
            ----------
            temporary_path : pathlib.Path
                与最终文件同目录的临时路径。

            Returns
            -------
            None
                写完根 metadata、通道、曲线及所有 Dataset 后返回。

            Workflow
            --------
            写公共 timestamps；逐通道和曲线创建编号 Group、UTF-8 attribute，
            分块写 values；仅独立时间轴曲线额外写 timestamps。
            """
            with h5py.File(temporary_path, "w") as hdf5_file:
                hdf5_file.attrs["schema_version"] = _CONTINUOUS_SCHEMA_VERSION
                _set_utf8_attribute(hdf5_file, "store_type", "continuous")

                if common_timestamps is not None:
                    _create_and_write_1d_dataset(
                        hdf5_file,
                        "common_timestamps",
                        common_timestamps,
                        chunk_points=chunk_points,
                        compression=compression,
                        compression_level=compression_level,
                        source_name="common_timestamps",
                        require_real=True,
                    )

                channels_group = hdf5_file.create_group("channels")
                channel_index = 0
                for channel_name, channel_curves in continuous_dict.items():
                    if channel_name == _COMMON_TIMESTAMPS_KEY:
                        continue
                    if not isinstance(channel_name, str) or not channel_name:
                        raise ValueError("通道名称必须是非空字符串。")
                    channel_group = channels_group.create_group(
                        f"channel_{channel_index:06d}"
                    )
                    _set_utf8_attribute(channel_group, "name", channel_name)
                    curves_group = channel_group.create_group("curves")

                    for curve_index, (curve_name, curve_data) in enumerate(
                            channel_curves.items()
                    ):
                        if not isinstance(curve_name, str) or not curve_name:
                            raise ValueError(
                                f"通道 {channel_name!r} 中的曲线名称必须是非空字符串。"
                            )
                        curve_group = curves_group.create_group(
                            f"curve_{curve_index:06d}"
                        )
                        _set_utf8_attribute(curve_group, "name", curve_name)

                        own_timestamps = curve_data.get("timestamps")
                        timestamp_source = "own" if own_timestamps is not None else "common"
                        _set_utf8_attribute(
                            curve_group,
                            "timestamp_source",
                            timestamp_source,
                        )
                        _create_and_write_1d_dataset(
                            curve_group,
                            "values",
                            curve_data["values"],
                            chunk_points=chunk_points,
                            compression=compression,
                            compression_level=compression_level,
                            source_name=(
                                f"通道 {channel_name!r} 的曲线 {curve_name!r} values"
                            ),
                        )
                        if timestamp_source == "own":
                            _create_and_write_1d_dataset(
                                curve_group,
                                "timestamps",
                                own_timestamps,
                                chunk_points=chunk_points,
                                compression=compression,
                                compression_level=compression_level,
                                source_name=(
                                    f"通道 {channel_name!r} 的曲线 "
                                    f"{curve_name!r} timestamps"
                                ),
                                require_real=True,
                            )
                    channel_index += 1
                hdf5_file.flush()

        target_path = _atomic_write_hdf5(
            _resolve_store_file_path(file_path, "continuous.h5"),
            writer,
            overwrite,
        )
        return cls(target_path, mode="r")

    def _load_and_validate_schema(self):
        """验证 continuous schema，建立名称索引但不读取完整曲线数组。

        Returns
        -------
        None
            成功后设置 common_timestamps、_channel_paths 和 _curve_paths。

        Workflow
        --------
        检查根 metadata 和公共时间轴；遍历编号 Group，解码真实名称，验证每条
        曲线的时间轴来源、Dataset 一维性、等长、chunk 和 numeric dtype。

        Raises
        ------
        ValueError
            schema、名称唯一性、路径、长度、chunk 或 dtype 不符合要求时抛出。
        """
        schema_version = self._file.attrs.get("schema_version")
        store_type = _decode_utf8(self._file.attrs.get("store_type", ""))
        if schema_version != _CONTINUOUS_SCHEMA_VERSION or store_type != "continuous":
            raise ValueError(
                f"不支持的 continuous HDF5 schema：version={schema_version!r}, "
                f"store_type={store_type!r}。"
            )
        if "channels" not in self._file:
            raise ValueError("continuous HDF5 缺少 /channels。")
        self.common_timestamps = self._file.get("common_timestamps")
        if self.common_timestamps is not None:
            if self.common_timestamps.ndim != 1:
                raise ValueError("/common_timestamps 必须是一维 Dataset。")
            if self.common_timestamps.chunks is None:
                raise ValueError("/common_timestamps 必须使用 chunked Dataset。")
            _validate_numeric_dtype(
                self.common_timestamps.dtype,
                "/common_timestamps",
                require_real=True,
            )

        self._channel_paths = {}
        self._curve_paths = {}
        for channel_internal_name in sorted(self._file["channels"].keys()):
            channel_group = self._file[f"channels/{channel_internal_name}"]
            if "name" not in channel_group.attrs or "curves" not in channel_group:
                raise ValueError(f"通道 Group {channel_internal_name!r} 结构不完整。")
            channel_name = _decode_utf8(channel_group.attrs["name"])
            if channel_name in self._channel_paths:
                raise ValueError(f"continuous HDF5 存在重复通道名称 {channel_name!r}。")
            self._channel_paths[channel_name] = channel_group.name
            self._curve_paths[channel_name] = {}

            for curve_internal_name in sorted(channel_group["curves"].keys()):
                curve_group = channel_group[f"curves/{curve_internal_name}"]
                if "name" not in curve_group.attrs or "values" not in curve_group:
                    raise ValueError(
                        f"曲线 Group {curve_group.name!r} 缺少 name 或 values。"
                    )
                curve_name = _decode_utf8(curve_group.attrs["name"])
                if curve_name in self._curve_paths[channel_name]:
                    raise ValueError(
                        f"通道 {channel_name!r} 中存在重复曲线名称 {curve_name!r}。"
                    )
                timestamp_source = _decode_utf8(
                    curve_group.attrs.get("timestamp_source", "")
                )
                if timestamp_source not in {"common", "own"}:
                    raise ValueError(
                        f"曲线 {channel_name!r}/{curve_name!r} 的 timestamp_source 无效。"
                    )
                if timestamp_source == "common":
                    if self.common_timestamps is None:
                        raise ValueError(
                            f"曲线 {channel_name!r}/{curve_name!r} 声明公共时间轴，"
                            "但文件缺少 /common_timestamps。"
                        )
                    timestamps_dataset = self.common_timestamps
                    if "timestamps" in curve_group:
                        raise ValueError("使用公共时间轴的曲线不能重复保存 timestamps。")
                else:
                    if "timestamps" not in curve_group:
                        raise ValueError(
                            f"曲线 {channel_name!r}/{curve_name!r} 缺少独立 timestamps。"
                        )
                    timestamps_dataset = curve_group["timestamps"]

                values_dataset = curve_group["values"]
                if values_dataset.ndim != 1 or timestamps_dataset.ndim != 1:
                    raise ValueError(
                        f"曲线 {channel_name!r}/{curve_name!r} 的 values 和 timestamps "
                        "必须是一维 Dataset。"
                    )
                if len(values_dataset) != len(timestamps_dataset):
                    raise ValueError(
                        f"曲线 {channel_name!r}/{curve_name!r} timestamps 长度为 "
                        f"{len(timestamps_dataset)}，values 长度为 {len(values_dataset)}。"
                    )
                if values_dataset.chunks is None or timestamps_dataset.chunks is None:
                    raise ValueError("continuous values 和 timestamps 必须使用 chunked Dataset。")
                _validate_numeric_dtype(
                    values_dataset.dtype,
                    f"曲线 {channel_name!r}/{curve_name!r} values",
                    require_real=False,
                )
                _validate_numeric_dtype(
                    timestamps_dataset.dtype,
                    f"曲线 {channel_name!r}/{curve_name!r} timestamps",
                    require_real=True,
                )
                self._curve_paths[channel_name][curve_name] = curve_group.name

    def _ensure_open(self):
        """确认 continuous HDF5 文件句柄仍有效。

        Returns
        -------
        None
            句柄有效时返回；关闭时抛出 RuntimeError。

        Workflow
        --------
        同时检查 `_file` 非 None 和 h5py id.valid。
        """
        if self._file is None or not self._file.id.valid:
            raise RuntimeError("ContinuousHDF5Store 已关闭。")

    def list_channels(self):
        """按文件中的稳定顺序返回真实通道名称。

        Returns
        -------
        list[str]
            所有通道的 UTF-8 原始名称；只复制轻量级名称列表。

        Workflow
        --------
        验证句柄后返回 `_channel_paths` 的插入顺序键列表，不读取曲线 Dataset。
        """
        self._ensure_open()
        return list(self._channel_paths)

    def list_curves(self, channel_name):
        """返回指定通道的真实曲线名称。

        Parameters
        ----------
        channel_name : str
            文件 attribute 中保存的通道真实名称。

        Returns
        -------
        list[str]
            该通道内全部曲线的真实名称，保持文件顺序。

        Workflow
        --------
        验证句柄和通道存在性，返回对应轻量级路径映射的键列表。
        """
        self._ensure_open()
        if channel_name not in self._curve_paths:
            raise KeyError(f"不存在通道 {channel_name!r}。")
        return list(self._curve_paths[channel_name])

    def get_curve_sources(self, channel_name, curve_name):
        """返回曲线的懒加载 Dataset 与时间轴来源。

        Parameters
        ----------
        channel_name : str
            通道真实名称。
        curve_name : str
            该通道内的曲线真实名称。

        Returns
        -------
        dict[str, object]
            包含 ``timestamps``（一维 h5py.Dataset）、``values``（等长一维
            h5py.Dataset）和 ``timestamp_source``（``"common"`` 或 ``"own"``）。

        Workflow
        --------
        通过名称索引定位曲线 Group，读取小型来源 attribute，选择公共或独立
        timestamps Dataset，并返回 Dataset 引用而不读取数据。
        """
        self._ensure_open()
        try:
            curve_path = self._curve_paths[channel_name][curve_name]
        except KeyError as exc:
            raise KeyError(f"不存在曲线 {channel_name!r}/{curve_name!r}。") from exc
        curve_group = self._file[curve_path]
        timestamp_source = _decode_utf8(curve_group.attrs["timestamp_source"])
        timestamps_dataset = (
            self.common_timestamps
            if timestamp_source == "common"
            else curve_group["timestamps"]
        )
        return {
            "timestamps": timestamps_dataset,
            "values": curve_group["values"],
            "timestamp_source": timestamp_source,
        }

    def get_curve_slice(
            self,
            channel_name,
            curve_name,
            start_timestamp,
            end_timestamp,
    ):
        """只读取一条曲线在指定闭区间真实时间窗口内的数据。

        Parameters
        ----------
        channel_name : str
            通道真实名称。
        curve_name : str
            曲线真实名称。
        start_timestamp : Real
            包含式时间窗口左边界。
        end_timestamp : Real
            包含式时间窗口右边界，必须不小于左边界。

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            当前窗口 timestamps 和 values，两者形状均为 ``(K,)``；无数据时 K=0。

        Workflow
        --------
        取得懒 Dataset，使用 O(log N) 标量二分求下标，再只读取两个对应切片。
        """
        sources = self.get_curve_sources(channel_name, curve_name)
        start_index, end_index = find_timestamp_slice(
            sources["timestamps"],
            start_timestamp,
            end_timestamp,
        )
        return (
            sources["timestamps"][start_index:end_index],
            sources["values"][start_index:end_index],
        )

    def flush(self):
        """刷新 continuous HDF5 缓冲区。

        Returns
        -------
        None
            flush 完成后返回。

        Workflow
        --------
        确认句柄有效，再调用 h5py flush；只读 Store 调用也安全。
        """
        self._ensure_open()
        self._file.flush()

    def close(self):
        """幂等关闭 continuous HDF5 文件句柄。

        Returns
        -------
        None
            已关闭或关闭完成均不返回数据。

        Workflow
        --------
        仅在句柄存在且有效时调用 close。
        """
        if self._file is not None and self._file.id.valid:
            self._file.close()

    def __enter__(self):
        """进入 with 上下文并返回当前 Store。

        Returns
        -------
        ContinuousHDF5Store
            当前已打开实例。

        Workflow
        --------
        检查句柄后返回 self。
        """
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出 with 上下文，关闭 Store 并保留原异常。

        Parameters
        ----------
        exc_type : type[BaseException] or None
            上下文中的异常类型。
        exc_value : BaseException or None
            异常实例。
        traceback : types.TracebackType or None
            异常回溯。

        Returns
        -------
        bool
            固定 False，不抑制异常。

        Workflow
        --------
        幂等关闭句柄，再返回 False。
        """
        self.close()
        return False


def plot_overview_waveform(ax, waveform, config):
    """在已有 Matplotlib Axes 上绘制一个 epoch waveform。

    函数只负责绘图，不读取 HDF5、不管理 Qt Widget，也不保存选择状态。默认使用
    waveform 的采样点位置作为横坐标；配置中提供 xlim 或 ylim 时再覆盖自动范围。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        需要绘图的 Axes。
    waveform : array-like
        已按单个 epoch 读取的一维波形。
    config : Mapping
        完整 config 或其 overview 区块。完整配置的默认来源和合并方式详见
        ``DataViewer.__init__`` 的 ``config_dict`` 参数。

    Returns
    -------
    matplotlib.lines.Line2D or None
        非空波形的折线对象；空波形返回 None。

    Raises
    ------
    ValueError
        waveform 不是一维数据，或 config 不是 Mapping 时抛出。

    Workflow
    --------
    1. 解析完整配置或 overview 子配置，验证当前单条 waveform 为一维。
    2. 清空 Axes 并设置轻量样式；非空数据以 ``arange(N)`` 为横坐标绘线。
    3. 空数据写提示文字，最后按配置覆盖 xlim/ylim 并返回折线对象。
    """
    if not isinstance(config, Mapping):
        raise ValueError("config 必须是字典或其他 Mapping 对象。")
    overview_config = config.get("overview", config)
    if not isinstance(overview_config, Mapping):
        raise ValueError("config['overview'] 必须是字典。")

    waveform_array = np.asarray(waveform)
    if waveform_array.ndim != 1:
        raise ValueError(
            f"Overview waveform 必须是一维数据，当前 shape 为 "
            f"{waveform_array.shape!r}。"
        )

    ax.clear()
    ax.set_facecolor("#fbfdff")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    line = None
    if len(waveform_array):
        sample_positions = np.arange(len(waveform_array))
        line, = ax.plot(
            sample_positions,
            waveform_array,
            color=overview_config["line_color"],
            linewidth=overview_config["line_width"],
        )
    else:
        ax.text(
            0.5,
            0.5,
            "waveform 为空",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#7890aa",
        )

    if overview_config["xlim"] is not None:
        ax.set_xlim(*overview_config["xlim"])
    if overview_config["ylim"] is not None:
        ax.set_ylim(*overview_config["ylim"])
    return line


class OverviewWidget(QFrame):
    """显示单个 epoch，并发送带原始 epoch 下标的鼠标事件。

    Widget 仅持有当前页面的一条 waveform；不访问 HDF5、不执行筛选或删除。
    ``clicked(int)`` 与 ``double_clicked(int)`` 信号中的整数始终是原始 epoch 下标。
    """

    clicked = pyqtSignal(int)
    double_clicked = pyqtSignal(int)

    def __init__(
            self,
            original_epoch_index,
            waveform,
            epoch_id,
            config,
            *,
            selected=False,
            parent=None,
    ):
        """创建一个只持有当前页单条 waveform 的 Overview 组件。

        Parameters
        ----------
        original_epoch_index : int
            HDF5 中稳定的原始 epoch 下标。
        waveform : array-like
            主窗口已按需读取的单条波形。
        epoch_id : str
            显示给用户的稳定 epoch ID。
        config : Mapping
            完整配置；本组件只读取 overview 区块。详见 ``DataViewer.__init__``
            的 ``config_dict`` 参数。
        selected : bool, optional
            初始选择状态。
        parent : QWidget or None, optional
            Qt 父组件。

        Returns
        -------
        None
            构造函数不返回值；成功后创建标签、Figure 和 Canvas。

        Workflow
        --------
        保存轻量状态，创建用于区分单击/双击的单次定时器，搭建 UI，再应用初始选择样式。
        """
        super().__init__(parent)
        self.original_epoch_index = int(original_epoch_index)
        self.waveform = waveform
        self.epoch_id = str(epoch_id)
        self.config = config
        self.is_selected = False
        self._plot_resources_released = False
        self._pending_click = False

        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(self._emit_pending_click)

        self._setup_ui()
        self.set_selected(selected)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def _setup_ui(self):
        """根据 Overview 配置创建 ID、选择标记和波形画布。

        Returns
        -------
        None
            完成布局后将 figure、canvas、ax 和标签保存为实例属性。

        Workflow
        --------
        建立纵向布局和标题行；创建单 Axes Figure；调用独立绘图函数；设置由配置
        决定的最小高度，最后把 Canvas 放入布局。
        """
        overview_config = self.config["overview"]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)

        header_layout = QHBoxLayout()
        self.epoch_id_label = QLabel(self.epoch_id)
        self.epoch_id_label.setStyleSheet("""
            font-weight: bold;
            font-size: 11px;
            background-color: #e5edff;
            color: #1e3a8a;
            padding: 4px 8px;
            border-radius: 12px;
        """)
        self.selected_icon = QLabel("✓ 已选择")
        self.selected_icon.setStyleSheet("""
            color: #b42318;
            font-size: 10px;
            font-weight: bold;
            background-color: #ffe4e1;
            padding: 2px 6px;
            border-radius: 10px;
        """)
        header_layout.addWidget(self.epoch_id_label)
        header_layout.addStretch()
        header_layout.addWidget(self.selected_icon)
        layout.addLayout(header_layout)

        self.figure = Figure(
            figsize=(1.2, 0.8),
            dpi=overview_config["figure_dpi"],
            facecolor="#fbfdff",
        )
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(int(overview_config["subplot_min_height"]))
        self.canvas.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.ax = self.figure.add_subplot(111)
        plot_overview_waveform(self.ax, self.waveform, self.config)
        self.figure.tight_layout(pad=0.3)
        self.canvas.draw()
        layout.addWidget(self.canvas)

        self.setMinimumWidth(120)
        self.setMinimumHeight(int(overview_config["subplot_min_height"]))

    def mousePressEvent(self, event):
        """处理鼠标按下，并延迟发送左键单击。

        Parameters
        ----------
        event : PyQt5.QtGui.QMouseEvent
            Qt 传入的鼠标事件。

        Returns
        -------
        None
            Qt 事件处理方法不返回业务数据。

        Workflow
        --------
        左键时记录 pending 并启动 200 ms 定时器，再交给父类执行默认处理。
        """
        if event.button() == Qt.LeftButton:
            self._pending_click = True
            self._click_timer.start(200)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """发送携带原始 epoch 下标的左键双击信号。

        Parameters
        ----------
        event : PyQt5.QtGui.QMouseEvent
            Qt 双击事件。

        Returns
        -------
        None
            通过 signal 输出下标，不直接返回值。

        Workflow
        --------
        左键双击时取消待发送单击、emit double_clicked、接受事件；其他键交给父类。
        """
        if event.button() == Qt.LeftButton:
            self._click_timer.stop()
            self._pending_click = False
            self.double_clicked.emit(self.original_epoch_index)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _emit_pending_click(self):
        """定时器到期后发送一次普通点击信号。

        Returns
        -------
        None
            通过 ``clicked(int)`` signal 输出，不返回业务值。

        Workflow
        --------
        仅当 pending 标记仍为 True 时清除标记并发送原始 epoch 下标。
        """
        if self._pending_click:
            self._pending_click = False
            self.clicked.emit(self.original_epoch_index)

    def set_selected(self, selected):
        """更新当前 Widget 的选择视觉状态。

        Parameters
        ----------
        selected : bool-like
            True 显示“已选择”和红色边框；False 恢复普通样式。

        Returns
        -------
        None
            只修改本 Widget 的显示状态。

        Workflow
        --------
        转为 bool，切换选择标签可见性，再设置选中或未选中的样式表。
        """
        self.is_selected = bool(selected)
        self.selected_icon.setVisible(self.is_selected)
        if self.is_selected:
            self.setStyleSheet("""
                QFrame {
                    border: 3px solid #e05a47;
                    border-radius: 10px;
                    background-color: #fff6f3;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    border: 1px solid #cbd8e6;
                    border-radius: 10px;
                    background-color: #ffffff;
                }
                QFrame:hover {
                    border: 2px solid #4da6a1;
                    background-color: #f2fbfa;
                }
            """)

    def release_plot_resources(self):
        """显式断开 Canvas 和 Figure，并释放当前页 waveform 引用。

        该方法可重复调用。翻页时主窗口会先调用它，再将 Widget 交给 Qt 延迟
        销毁，避免旧 Axes 和 waveform 继续被引用。

        Returns
        -------
        None
            释放完成或此前已释放时均不返回数据。

        Workflow
        --------
        停止点击定时器；清空并断开 Figure；关闭、脱离并延迟销毁 Canvas；最后
        将 Axes、Canvas、Figure 和 waveform 引用设为 None。
        """
        if self._plot_resources_released:
            return
        self._plot_resources_released = True
        self._click_timer.stop()
        self._pending_click = False

        old_canvas = self.canvas
        old_figure = self.figure
        if old_canvas is not None and hasattr(old_canvas, "_draw_pending"):
            old_canvas._draw_pending = False
        if old_figure is not None:
            old_figure.clear()
            old_figure.set_canvas(None)
        if old_canvas is not None:
            old_canvas.close()
            old_canvas.setParent(None)
            old_canvas.deleteLater()

        self.ax = None
        self.canvas = None
        self.figure = None
        self.waveform = None

    def closeEvent(self, event):
        """单独关闭组件时也释放 Matplotlib 资源。

        Parameters
        ----------
        event : PyQt5.QtGui.QCloseEvent
            Qt 关闭事件。

        Returns
        -------
        None
            释放后把事件交给父类处理。

        Workflow
        --------
        幂等调用 ``release_plot_resources``，再调用 QFrame.closeEvent。
        """
        self.release_plot_resources()
        super().closeEvent(event)


def plot_detail_channel(
        ax,
        channel_name,
        curves,
        epoch_center_timestamp,
        start_timestamp,
        end_timestamp,
        config,
        *,
        target_width,
        epoch_region_start=None,
        epoch_region_end=None,
):
    """在一个 Axes 中绘制一个通道的全部曲线。

    本函数不访问 HDF5，也不管理 Qt 窗口或通道分页。调用方只需要传入当前
    时间窗口已经读取到的曲线切片。每条非空曲线在交给 Matplotlib 前都会使用
    min-max envelope 下采样，从而保留尖峰并限制绘图点数。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        需要绘制当前通道的 Axes。
    channel_name : str
        当前通道的真实名称。
    curves : Mapping
        曲线名称到包含 timestamps 和 values 的 Mapping。
    epoch_center_timestamp : Real or None
        当前 epoch 片段的时间中点，用深红色虚线标记。它不是当前界面横轴的
        中点；没有已定位 epoch 时传 None，不绘制虚线。
    start_timestamp : Real
        当前显示窗口开始时间。
    end_timestamp : Real
        当前显示窗口结束时间。
    config : Mapping
        完整 config 或 detail 配置区块。完整配置说明详见
        ``DataViewer.__init__`` 的 ``config_dict`` 参数。
    target_width : Real
        当前 Canvas 可用于下采样的像素宽度。
    epoch_region_start : Real or None, keyword-only
        当前 epoch 片段开始时间；与 epoch_region_end 同时提供时绘制红色区域。
    epoch_region_end : Real or None, keyword-only
        当前 epoch 片段结束时间，必须不小于 epoch_region_start。

    Returns
    -------
    dict
        曲线名称到 Matplotlib Line2D 的映射；无数据的曲线不会出现在结果中。

    Raises
    ------
    ValueError
        参数类型、时间范围或曲线切片结构无效时抛出。

    Workflow
    --------
    1. 解析 detail 配置，验证显示窗口、epoch 中点/区域和曲线 Mapping。
    2. 先添加 epoch 红色区域；每条曲线验证等长一维切片，下采样后绘制。
    3. 添加 epoch 中点虚线、xlim、标题、网格、通道优先 ylim 和可选图例。
    """
    if not isinstance(config, Mapping):
        raise ValueError("config 必须是字典或其他 Mapping 对象。")
    detail_config = config.get("detail", config)
    if not isinstance(detail_config, Mapping):
        raise ValueError("config['detail'] 必须是字典。")
    if not isinstance(channel_name, str) or not channel_name:
        raise ValueError("channel_name 必须是非空字符串。")
    if not isinstance(curves, Mapping):
        raise ValueError("curves 必须是曲线名称到数据切片的 Mapping。")
    for field_name, value in (
            ("start_timestamp", start_timestamp),
            ("end_timestamp", end_timestamp),
    ):
        if not _is_real_scalar(value):
            raise ValueError(f"{field_name} 必须是有限实数。")
    if (
            epoch_center_timestamp is not None
            and not _is_real_scalar(epoch_center_timestamp)
    ):
        raise ValueError("epoch_center_timestamp 必须是有限实数或 None。")
    if (epoch_region_start is None) != (epoch_region_end is None):
        raise ValueError("epoch_region_start 和 epoch_region_end 必须同时提供或同时为 None。")
    if epoch_region_start is not None:
        if (
                not _is_real_scalar(epoch_region_start)
                or not _is_real_scalar(epoch_region_end)
                or epoch_region_start > epoch_region_end
        ):
            raise ValueError("epoch 标记区域必须是 start <= end 的有限实数范围。")
    if start_timestamp > end_timestamp:
        raise ValueError("start_timestamp 不能大于 end_timestamp。")

    ax.clear()
    if epoch_region_start is not None:
        ax.axvspan(
            epoch_region_start,
            epoch_region_end,
            alpha=0.24,
            color="#f26b5b",
            zorder=0,
        )
    plotted_lines = {}
    for curve_name, curve_data in curves.items():
        if not isinstance(curve_data, Mapping):
            raise ValueError(f"曲线 {curve_name!r} 必须是包含切片数据的 Mapping。")
        if "timestamps" not in curve_data or "values" not in curve_data:
            raise ValueError(f"曲线 {curve_name!r} 必须包含 timestamps 和 values。")

        timestamps = curve_data["timestamps"]
        values = curve_data["values"]
        timestamps_array = np.asarray(timestamps)
        values_array = np.asarray(values)
        if timestamps_array.ndim != 1 or values_array.ndim != 1:
            raise ValueError(f"曲线 {curve_name!r} 的 timestamps 和 values 必须是一维。")
        if len(timestamps_array) != len(values_array):
            raise ValueError(
                f"曲线 {curve_name!r} timestamps 长度为 {len(timestamps_array)}，"
                f"values 长度为 {len(values_array)}，二者必须相同。"
            )
        if len(values_array) == 0:
            continue

        plot_timestamps, plot_values = downsample_min_max(
            timestamps_array,
            values_array,
            target_width,
            detail_config["max_points_per_pixel"],
        )
        line, = ax.plot(
            plot_timestamps,
            plot_values,
            linewidth=detail_config["line_width"],
            label=str(curve_name),
        )
        plotted_lines[curve_name] = line

    if not plotted_lines:
        ax.text(
            0.5,
            0.5,
            "当前时间范围无数据",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#6b7280",
        )

    if epoch_center_timestamp is not None:
        ax.axvline(
            x=epoch_center_timestamp,
            color="#a83232",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            zorder=1,
            label="_epoch_center_timestamp",
        )
    ax.set_xlim(start_timestamp, end_timestamp)
    ax.set_title(channel_name, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("幅值")
    ax.grid(True, alpha=0.25, linestyle="--")

    channel_ylims = detail_config["channel_ylims"]
    selected_ylim = channel_ylims.get(channel_name, detail_config["ylim"])
    if selected_ylim is not None:
        ax.set_ylim(*selected_ylim)

    if detail_config["show_legend"] and plotted_lines:
        ax.legend(loc="upper right")
    return plotted_lines


class DetailChartWindow(QMainWindow):
    """按真实时间窗口懒加载 continuous 数据的唯一 Detail 窗口。

    一个实例管理全部动态通道分页。每次只读取当前通道页、当前真实时间范围，
    最近切片保存在容量受配置限制的 LRU 缓存中。
    """

    _RESERVED_VERTICAL_PIXELS = 170
    _MAX_QT_SLIDER_VALUE = 1_000_000_000

    def __init__(self, continuous_store, config, parent=None, *, epochs_store=None):
        """创建动态通道分页窗口，但在收到时间戳前不读取曲线。

        Parameters
        ----------
        continuous_store : ContinuousHDF5Store
            已打开且在 Detail 使用期间保持有效的连续数据 Store。
        config : Mapping
            已合并并验证的完整配置。默认来源和合并规则详见
            ``DataViewer.__init__`` 的 ``config_dict`` 参数。
        parent : QWidget or None, optional
            通常是唯一的 DataViewer 主窗口。
        epochs_store : EpochsHDF5Store or None, keyword-only
            主窗口持有的 epoch Store，用于按原始下标/ID 跳转以及上一个、下一个
            epoch。传 None 时连续数据和公共时间轴滑块仍可使用，但波形导航禁用。

        Raises
        ------
        ValueError
            Store 类型或配置无效时抛出。

        Returns
        -------
        None
            构造函数不返回值；成功后窗口尚无中心时间，不会读取曲线切片。

        Workflow
        --------
        验证 Store/配置，缓存通道名称和时间设置，初始化空 LRU 缓存，搭建 UI，
        根据窗口高度计算每页通道数，并绘制“请选择 epoch”的初始页面。
        """
        super().__init__(parent)
        if not isinstance(continuous_store, ContinuousHDF5Store):
            raise ValueError(
                "DetailChartWindow 的 continuous_store 必须是 ContinuousHDF5Store。"
            )
        continuous_store._ensure_open()
        if not isinstance(config, Mapping):
            raise ValueError("DetailChartWindow 的 config 必须是 Mapping。")
        if epochs_store is not None:
            if not isinstance(epochs_store, EpochsHDF5Store):
                raise ValueError(
                    "DetailChartWindow 的 epochs_store 必须是 "
                    "EpochsHDF5Store 或 None。"
                )
            epochs_store._ensure_open()

        self.continuous_store = continuous_store
        self.epochs_store = epochs_store
        self.config = config
        self.detail_config = config["detail"]
        self.channel_names = self.continuous_store.list_channels()
        self.channel_visibility = {}
        self.curve_visibility = {}
        self._apply_plot_visibility_from_config()

        self.current_timestamp = None
        self.start_timestamp = None
        self.end_timestamp = None
        self.xlim = None
        self.time_before = float(self.detail_config["time_before"])
        self.time_after = float(self.detail_config["time_after"])
        self.current_epoch_index = None
        self.epoch_region_start = None
        self.epoch_region_end = None
        self.epoch_center_timestamp = None
        self._epoch_id_to_index = (
            {
                epoch_id: epoch_index
                for epoch_index, epoch_id in enumerate(self.epochs_store.epochs_ids)
            }
            if self.epochs_store is not None
            else {}
        )

        self._common_timestamp_count = (
            len(self.continuous_store.common_timestamps)
            if self.continuous_store.common_timestamps is not None
            else 0
        )
        self._slider_maximum = min(
            max(0, self._common_timestamp_count - 1),
            self._MAX_QT_SLIDER_VALUE,
        )
        self._syncing_time_slider = False

        self.current_channel_page = 0
        self.channels_per_page = 1
        self.total_channel_pages = 1
        self.current_channel_names = []
        self._curve_settings_dialog = None
        self._curve_settings_channel_checks = {}
        self._curve_settings_curve_checks = {}
        self.axes = []
        self.last_loaded_curve_keys = []
        # 缓存只保存有限数量的“单曲线、单时间窗”切片，容量由 config 控制。
        # OrderedDict 让最久未使用的切片可以在 O(1) 时间内被淘汰。
        self._slice_cache = OrderedDict()
        self._slider_update_timer = QTimer(self)
        self._slider_update_timer.setSingleShot(True)
        self._slider_update_timer.setInterval(120)
        self._slider_update_timer.timeout.connect(
            self._apply_common_timestamp_slider_position
        )
        self._slider_motion_timer = QTimer(self)
        self._slider_motion_timer.setInterval(30)
        self._slider_motion_timer.timeout.connect(self._advance_slider_motion)
        self._slider_motion_direction = 0
        self._slider_motion_last_time = None
        self._slider_motion_last_redraw = None
        self._slider_motion_fraction = 0.0
        self._slider_motion_moved = False
        self._ui_ready = False
        self._plot_resources_released = False

        self.setWindowTitle("Continuous Detail - 按真实时间戳查看")
        self.setStyleSheet(APP_STYLE_SHEET)
        self.resize(
            int(self.detail_config["window_width"]),
            int(self.detail_config["window_height"]),
        )
        self._init_ui()
        self.channels_per_page = self.calculate_channels_per_page()
        self._update_total_channel_pages()
        self._ui_ready = True
        self.load_channel_page()

    def _apply_plot_visibility_from_config(self):
        """按配置文件中的勾选状态初始化当前通道和曲线可见性。

        未记录的通道或曲线默认启用；配置中已经不存在的旧名称会被忽略。
        该方法只处理名称和布尔值，不读取任何曲线数据。
        """
        visibility = self.detail_config.get("plot_visibility", {})
        channel_values = visibility.get("channels", {})
        curve_values = visibility.get("curves", {})
        self.channel_visibility = {
            channel_name: bool(channel_values.get(channel_name, True))
            for channel_name in self.channel_names
        }
        self.curve_visibility = {}
        for channel_name in self.channel_names:
            self.curve_visibility[channel_name] = {
                curve_name: bool(
                    curve_values.get(channel_name, {}).get(curve_name, True)
                )
                for curve_name in self.continuous_store.list_curves(channel_name)
            }

    def _reload_plot_visibility_from_json(self):
        """每次打开曲线设置前从工作目录重新读取最新勾选状态。"""
        parent = self.parent()
        config_path = getattr(parent, "config_path", None)
        if config_path is None or not Path(config_path).exists():
            self._apply_plot_visibility_from_config()
            return
        try:
            latest_config = load_config_json(config_path)
            self.detail_config["plot_visibility"] = deepcopy(
                latest_config["detail"].get("plot_visibility", {})
            )
            self._apply_plot_visibility_from_config()
        except (OSError, ValueError, KeyError):
            # 配置文件暂时不可读时保留内存状态，避免设置按钮阻断绘图。
            self._apply_plot_visibility_from_config()

    def _enabled_curve_names(self, channel_name):
        """返回一个通道当前勾选的曲线名称列表。"""
        return [
            curve_name
            for curve_name, enabled in self.curve_visibility.get(channel_name, {}).items()
            if enabled
        ]

    def _visible_channel_names(self):
        """返回勾选启用的 Detail 子图名称列表。"""
        return [
            channel_name
            for channel_name in self.channel_names
            if self.channel_visibility.get(channel_name, True)
        ]

    def _show_curve_settings(self):
        """打开曲线/子图勾选窗口，并在打开瞬间刷新 JSON 状态。"""
        self._reload_plot_visibility_from_json()
        dialog = QDialog(self)
        dialog.setWindowTitle("设置 Detail 曲线")
        dialog.resize(900, 620)
        dialog.setStyleSheet(APP_STYLE_SHEET)
        self._curve_settings_dialog = dialog
        self._curve_settings_channel_checks = {}
        self._curve_settings_curve_checks = {}

        dialog_layout = QVBoxLayout(dialog)
        hint = QLabel("取消子图或曲线后，点击保存即可应用；未勾选曲线的子图会显示提示。")
        hint.setStyleSheet("color: #0f766e; padding: 4px;")
        dialog_layout.addWidget(hint)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 2)
        grid.addWidget(QLabel("Detail 子图"), 0, 0)
        grid.addWidget(QLabel("所属曲线（右侧 2/3）"), 0, 1)
        row = 1
        for channel_name in self.channel_names:
            curve_names = list(self.curve_visibility.get(channel_name, {}))
            span = max(1, len(curve_names))
            channel_check = QCheckBox(channel_name)
            channel_check.setChecked(self.channel_visibility.get(channel_name, True))
            grid.addWidget(channel_check, row, 0, span, 1, Qt.AlignTop)
            self._curve_settings_channel_checks[channel_name] = channel_check
            if not curve_names:
                grid.addWidget(QLabel("（无曲线）"), row, 1)
                row += 1
                continue
            for curve_name in curve_names:
                curve_check = QCheckBox(curve_name)
                curve_check.setChecked(
                    self.curve_visibility[channel_name].get(curve_name, True)
                )
                grid.addWidget(curve_check, row, 1)
                self._curve_settings_curve_checks[(channel_name, curve_name)] = curve_check
                row += 1
        scroll.setWidget(content)
        dialog_layout.addWidget(scroll, 1)
        button_row = QHBoxLayout()
        button_row.addStretch()
        cancel_button = QPushButton("取消")
        save_button = QPushButton("保存")
        cancel_button.clicked.connect(dialog.reject)
        save_button.clicked.connect(
            lambda _checked=False: self._save_curve_settings(dialog)
        )
        button_row.addWidget(cancel_button)
        button_row.addWidget(save_button)
        dialog_layout.addLayout(button_row)
        dialog.setModal(False)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _save_curve_settings(self, dialog):
        """保存曲线/子图勾选状态到全局 config.json 并刷新当前页面。"""
        channel_visibility = {
            channel_name: check.isChecked()
            for channel_name, check in self._curve_settings_channel_checks.items()
        }
        curve_visibility = {}
        for (channel_name, curve_name), check in self._curve_settings_curve_checks.items():
            curve_visibility.setdefault(channel_name, {})[curve_name] = check.isChecked()
        plot_visibility = {
            "channels": channel_visibility,
            "curves": curve_visibility,
        }
        self.channel_visibility = channel_visibility
        self.curve_visibility = {
            channel_name: {
                curve_name: curve_visibility.get(channel_name, {}).get(curve_name, True)
                for curve_name in self.continuous_store.list_curves(channel_name)
            }
            for channel_name in self.channel_names
        }
        self.detail_config["plot_visibility"] = deepcopy(plot_visibility)
        parent = self.parent()
        config_path = getattr(parent, "config_path", None)
        if config_path is not None:
            save_config_json_atomic(config_path, self.config)
        self.current_channel_page = 0
        self.load_channel_page()
        dialog.accept()

    def _init_ui(self):
        """创建三行 Detail 控制区、Matplotlib 画布和通道翻页控件。

        Returns
        -------
        None
            将输入框、按钮、Figure、Canvas、toolbar 和页标签保存为实例属性。

        Workflow
        --------
        第一行按等宽两栏放置 epoch 信息和波形导航；第二行放置公共时间轴滑块及
        纵轴输入；第三行单独显示公共时间轴位置和时间；随后创建可重建 Axes 的
        Figure/Canvas，并连接通道上一页/下一页按钮。
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        top_controls = QVBoxLayout()
        top_controls.setContentsMargins(0, 0, 0, 0)
        top_controls.setSpacing(7)

        # 左半栏只显示可稳定辨认当前波形的原始下标和 ID。标签允许鼠标选择文本，
        # 用户遇到数百万级下标或较长 ID 时可以直接复制，而无需从图中抄写。
        self.timestamp_label = QLabel(
            "epoch index：尚未选择    epoch id：尚未选择"
        )
        self.timestamp_label.setObjectName("epochInfoLabel")
        self.timestamp_label.setStyleSheet(
            "font-weight: 600; color: #1e3a5f; background-color: #e8f0fb;"
            "border: 1px solid #c8d8ea; border-radius: 6px; padding: 6px 10px;"
        )
        self.timestamp_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.timestamp_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        self.timestamp_label.setMinimumWidth(430)
        first_row = QHBoxLayout()
        first_row.setContentsMargins(0, 0, 0, 0)
        first_row.setSpacing(14)
        first_row.addWidget(self.timestamp_label, 1)

        # 右半栏把额外宽度全部交给输入框。按钮保留 Qt 根据文字计算的原始尺寸，
        # 因而窗口变宽时只有输入框随之增长，长 epoch ID 不易被截断。
        epoch_navigation_widget = QWidget()
        epoch_navigation_layout = QHBoxLayout(epoch_navigation_widget)
        epoch_navigation_layout.setContentsMargins(0, 0, 0, 0)
        epoch_navigation_layout.setSpacing(6)
        epoch_navigation_layout.addWidget(QLabel("波形下标或 ID："))
        self.epoch_jump_input = QLineEdit()
        self.epoch_jump_input.setPlaceholderText("例如 12 或 epoch_12")
        self.epoch_jump_input.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        epoch_navigation_layout.addWidget(self.epoch_jump_input, 1)
        self.jump_to_epoch_btn = QPushButton("波形跳转")
        self.previous_epoch_btn = QPushButton("◀ 上一个波形")
        self.next_epoch_btn = QPushButton("下一个波形 ▶")
        self.jump_to_epoch_btn.clicked.connect(self.jump_to_epoch_from_input)
        self.previous_epoch_btn.clicked.connect(self.previous_epoch)
        self.next_epoch_btn.clicked.connect(self.next_epoch)
        epoch_navigation_layout.addWidget(self.jump_to_epoch_btn)
        epoch_navigation_layout.addWidget(self.previous_epoch_btn)
        epoch_navigation_layout.addWidget(self.next_epoch_btn)
        first_row.addWidget(epoch_navigation_widget, 1)
        top_controls.addLayout(first_row)

        # 公共时间轴占第二行左半栏。当前位置和时间移到第三行后，滑块能得到更多
        # 水平空间，在超长时间轴中拖动时更容易进行较细粒度的定位。
        slider_widget = QWidget()
        slider_layout = QHBoxLayout(slider_widget)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(6)
        slider_layout.addWidget(QLabel("公共时间轴："))
        self.move_time_left_btn = QPushButton("◀")
        self.move_time_left_btn.setToolTip("按住后按设置速度向左移动公共时间轴")
        self.move_time_left_btn.setMaximumWidth(42)
        self.move_time_left_btn.setEnabled(self._common_timestamp_count > 0)
        self.move_time_left_btn.pressed.connect(
            lambda: self._start_slider_motion(-1)
        )
        self.move_time_left_btn.released.connect(self._stop_slider_motion)
        slider_layout.addWidget(self.move_time_left_btn)
        self.common_timestamp_slider = QSlider(Qt.Horizontal)
        self.common_timestamp_slider.setRange(0, self._slider_maximum)
        self.common_timestamp_slider.setSingleStep(1)
        self.common_timestamp_slider.setPageStep(
            max(1, self._slider_maximum // 100)
        )
        self.common_timestamp_slider.setEnabled(self._common_timestamp_count > 0)
        self.common_timestamp_slider.valueChanged.connect(
            self._on_common_timestamp_slider_changed
        )
        self.common_timestamp_slider.sliderMoved.connect(
            self._preview_common_timestamp_slider_position
        )
        slider_layout.addWidget(self.common_timestamp_slider, 1)
        self.move_time_right_btn = QPushButton("▶")
        self.move_time_right_btn.setToolTip("按住后按设置速度向右移动公共时间轴")
        self.move_time_right_btn.setMaximumWidth(42)
        self.move_time_right_btn.setEnabled(self._common_timestamp_count > 0)
        self.move_time_right_btn.pressed.connect(
            lambda: self._start_slider_motion(1)
        )
        self.move_time_right_btn.released.connect(self._stop_slider_motion)
        slider_layout.addWidget(self.move_time_right_btn)
        second_row = QHBoxLayout()
        second_row.setContentsMargins(0, 0, 0, 0)
        second_row.setSpacing(14)
        second_row.addWidget(slider_widget, 3)

        # 纵轴区使用第二行右半栏，不再显示重复的“统一纵轴”标题。两个输入框均
        # 可随窗口伸展；应用按钮使用紧凑宽度，让数值字段获得主要空间。
        ylim_widget = QWidget()
        ylim_layout = QHBoxLayout(ylim_widget)
        ylim_layout.setContentsMargins(0, 0, 0, 0)
        ylim_layout.setSpacing(6)
        self.ylim_min_input = QLineEdit()
        self.ylim_min_input.setPlaceholderText("最小值")
        self.ylim_min_input.setMinimumWidth(120)
        self.ylim_min_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ylim_max_input = QLineEdit()
        self.ylim_max_input.setPlaceholderText("最大值")
        self.ylim_max_input.setMinimumWidth(120)
        self.ylim_max_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.apply_ylim_btn = QPushButton("应用纵轴")
        self.apply_ylim_btn.setMaximumWidth(88)
        self.apply_ylim_btn.clicked.connect(self.apply_uniform_ylim_from_inputs)
        ylim_layout.addWidget(self.ylim_min_input, 1)
        ylim_layout.addWidget(QLabel("至"))
        ylim_layout.addWidget(self.ylim_max_input, 1)
        ylim_layout.addWidget(self.apply_ylim_btn)
        second_row.addWidget(ylim_widget, 1)
        top_controls.addLayout(second_row)

        # 第三行从最左侧显示当前位置、公共时间轴总长度和五位有效数字时间。
        # 最小宽度按十亿级下标的典型文本计算，避免窗口布局因数字增长而跳动。
        self.common_timestamp_slider_label = QLabel()
        position_width_sample = (
            "当前位置 / 总长度：1,000,000,000 / 1,000,000,000    "
            "时间：-1.2346e+05 s"
        )
        self.common_timestamp_slider_label.setMinimumWidth(
            self.common_timestamp_slider_label.fontMetrics().horizontalAdvance(
                position_width_sample
            ) + 12
        )
        self.common_timestamp_slider_label.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter
        )
        self.curve_settings_btn = QPushButton("设置曲线")
        self.curve_settings_btn.setObjectName("curveSettingsButton")
        self.curve_settings_btn.clicked.connect(self._show_curve_settings)
        third_row = QHBoxLayout()
        third_row.setContentsMargins(0, 0, 0, 0)
        third_row.setSpacing(10)
        third_row.addWidget(self.common_timestamp_slider_label, 1)
        third_row.addWidget(self.curve_settings_btn)
        top_controls.addLayout(third_row)
        layout.addLayout(top_controls)

        configured_ylim = self.detail_config["ylim"]
        if configured_ylim is not None:
            self.ylim_min_input.setText(str(configured_ylim[0]))
            self.ylim_max_input.setText(str(configured_ylim[1]))

        navigation_available = (
            self.epochs_store is not None
            and self.epochs_store.start_timestamps is not None
            and len(self.epochs_store) > 0
        )
        self.epoch_jump_input.setEnabled(navigation_available)
        self.jump_to_epoch_btn.setEnabled(navigation_available)
        self.previous_epoch_btn.setEnabled(navigation_available)
        self.next_epoch_btn.setEnabled(navigation_available)

        self._update_common_timestamp_slider_label()

        self.figure = Figure(
            figsize=(
                max(1.0, float(self.detail_config["figure_width"]) / 100.0),
                max(1.0, float(self.detail_config["subplot_height"]) / 100.0),
            ),
            dpi=100,
            facecolor="#fbfdff",
        )
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setFixedWidth(
            max(1, int(round(float(self.detail_config["figure_width"]))))
        )
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        page_controls = QHBoxLayout()
        self.previous_channel_page_btn = QPushButton("◀ 上一页通道")
        self.next_channel_page_btn = QPushButton("下一页通道 ▶")
        self.channel_page_label = QLabel()
        self.channel_page_label.setAlignment(Qt.AlignCenter)
        self.channel_page_label.setStyleSheet("font-weight: bold;")
        self.previous_channel_page_btn.clicked.connect(self.previous_channel_page)
        self.next_channel_page_btn.clicked.connect(self.next_channel_page)
        page_controls.addStretch()
        page_controls.addWidget(self.previous_channel_page_btn)
        page_controls.addWidget(self.channel_page_label)
        page_controls.addWidget(self.next_channel_page_btn)
        page_controls.addStretch()
        layout.addLayout(page_controls)

    def calculate_channels_per_page(self, available_height=None):
        """根据 Detail 可用高度和单个子图高度计算每页通道数量。

        Parameters
        ----------
        available_height : int or None, optional
            显式给定的绘图区域高度。None 时用窗口高度减去工具栏、时间控制和
            翻页区域的保守预留高度。

        Returns
        -------
        int
            每页至少显示一个通道。

        Workflow
        --------
        未显式给高度时从当前窗口高度减去固定控件预留；再用可用高度整除
        ``subplot_height``，并把结果限制为至少 1。
        """
        if available_height is None:
            available_height = max(
                1,
                int(self.height()) - self._RESERVED_VERTICAL_PIXELS,
            )
        subplot_height = float(self.detail_config["subplot_height"])
        return max(1, int(float(available_height) // subplot_height))

    def _update_total_channel_pages(self):
        """根据通道总数和当前每页容量更新分页边界。

        Returns
        -------
        None
            更新 ``total_channel_pages`` 和 ``current_channel_page``。

        Workflow
        --------
        用向上取整计算总页数（空通道也保留一页），再把当前页夹在合法范围内。
        """
        visible_channel_count = len(self._visible_channel_names())
        self.total_channel_pages = max(
            1,
            (visible_channel_count + self.channels_per_page - 1)
            // self.channels_per_page,
        )
        self.current_channel_page = min(
            max(0, self.current_channel_page),
            self.total_channel_pages - 1,
        )

    def _calculate_epoch_time_region(self, epoch_index, start_timestamp):
        """根据 waveform 点数和公共时间轴计算 epoch 片段终点与中点。

        由于输入结构只给出 start_timestamp，没有单独保存 epoch 结束时间，本函数
        假定 waveform 的每个点与 common_timestamps 的一个连续采样点对应。

        Parameters
        ----------
        epoch_index : Integral
            原始 epoch 下标。
        start_timestamp : Real
            该 epoch 在连续时间坐标中的真实起始时间。

        Returns
        -------
        tuple[float, float, float]
            ``(region_start, region_end, center)``。没有公共时间轴或 waveform 为空
            时三者都退化为 start_timestamp，因此仍可绘制中心虚线。

        Workflow
        --------
        通过 offsets 相减取得 waveform 长度，不读取波形；在公共时间轴上用标量
        二分找到首个不早于 start 的点；向后移动 ``waveform_length - 1`` 个点并
        限制到时间轴末尾，只读取终点一个标量；最后计算算术中点。
        """
        region_start = float(start_timestamp)
        if (
                self.epochs_store is None
                or self.continuous_store.common_timestamps is None
        ):
            return region_start, region_start, region_start

        waveform_length = int(
            self.epochs_store.offsets[epoch_index + 1]
            - self.epochs_store.offsets[epoch_index]
        )
        if waveform_length <= 1 or self._common_timestamp_count == 0:
            return region_start, region_start, region_start

        timestamps = self.continuous_store.common_timestamps
        left = 0
        right = self._common_timestamp_count
        while left < right:
            middle = (left + right) // 2
            if timestamps[middle] < region_start:
                left = middle + 1
            else:
                right = middle
        start_index = min(left, self._common_timestamp_count - 1)
        end_index = min(
            start_index + waveform_length - 1,
            self._common_timestamp_count - 1,
        )
        region_end_value = timestamps[end_index]
        region_end = float(
            region_end_value.item()
            if hasattr(region_end_value, "item")
            else region_end_value
        )
        region_end = max(region_start, region_end)
        return region_start, region_end, (region_start + region_end) / 2.0

    def set_current_timestamp(
            self,
            current_timestamp,
            *,
            epoch_index=None,
            preserve_epoch=False,
    ):
        """使用 epoch 的真实时间戳建立新 Detail 时间窗口。

        每次双击新的 epoch 都从 config 的 time_before/time_after 重新计算窗口，
        并回到通道第一页。之后只切换通道页时不会修改这些时间状态。

        Parameters
        ----------
        current_timestamp : Real
            epochs Store 中 start_timestamps[original_epoch_index] 的标量值。
        epoch_index : Integral or None, keyword-only
            时间来自某个 epoch 时传其原始下标；滑块自由跳转时传 None。该值仅
            决定波形导航状态，不改变时间范围计算。
        preserve_epoch : bool, keyword-only
            True 表示只移动视图中心，保留现有 epoch 红色区域和中点虚线；公共
            时间轴滑块使用此模式。False 且 epoch_index=None 时清除 epoch 标记。

        Returns
        -------
        None
            更新时间状态和窗口显示后返回。

        Workflow
        --------
        验证有限实数和可选下标；按 preserve_epoch 更新或保留 epoch 区域；按
        time_before/time_after 计算视图闭区间；同步滑块并只加载当前通道页。
        """
        if not _is_real_scalar(current_timestamp):
            raise ValueError("current_timestamp 必须是有限实数。")
        if epoch_index is not None:
            if self.epochs_store is None:
                raise ValueError("没有 epochs Store，不能设置 epoch_index。")
            epoch_index = _normalize_indices(
                epoch_index,
                len(self.epochs_store),
                "epoch_index",
            )[0]
        if not preserve_epoch:
            self.current_epoch_index = epoch_index
            if epoch_index is None:
                self.epoch_region_start = None
                self.epoch_region_end = None
                self.epoch_center_timestamp = None
            else:
                (
                    self.epoch_region_start,
                    self.epoch_region_end,
                    self.epoch_center_timestamp,
                ) = self._calculate_epoch_time_region(
                    epoch_index,
                    current_timestamp,
                )
        self.current_timestamp = float(current_timestamp)
        self.start_timestamp = self.current_timestamp - self.time_before
        self.end_timestamp = self.current_timestamp + self.time_after
        self.xlim = (self.start_timestamp, self.end_timestamp)
        self.current_channel_page = 0
        epoch_text = "epoch index：尚未选择    epoch id：尚未选择"
        if self.current_epoch_index is not None:
            epoch_text = (
                f"epoch index：{self.current_epoch_index:,}    "
                f"epoch id：{self.epochs_store.epochs_ids[self.current_epoch_index]}"
            )
        self.timestamp_label.setText(epoch_text)
        common_index = self._nearest_common_timestamp_index(self.current_timestamp)
        if common_index is not None:
            self._syncing_time_slider = True
            try:
                self.common_timestamp_slider.setValue(
                    self._common_index_to_slider_value(common_index)
                )
            finally:
                self._syncing_time_slider = False
            self._update_common_timestamp_slider_label()
        self._update_epoch_navigation_buttons()
        self.load_channel_page()

    def _slider_value_to_common_index(self, slider_value):
        """把 Qt 滑块整数位置映射到公共时间轴原始下标。

        Parameters
        ----------
        slider_value : Integral
            范围为 ``[0, self._slider_maximum]`` 的 Qt 滑块值。

        Returns
        -------
        int or None
            范围为 ``[0, common_timestamp_count - 1]`` 的公共时间轴下标；没有
            公共时间轴时返回 None。

        Workflow
        --------
        公共时间轴长度不超过十亿点时一一对应；超过十亿点时按两端严格对齐的
        比例换算并四舍五入。整个过程只使用长度和整数，不读取 timestamps。
        """
        if self._common_timestamp_count <= 0:
            return None
        if self._slider_maximum <= 0:
            return 0
        bounded_value = min(max(0, int(slider_value)), self._slider_maximum)
        if self._common_timestamp_count - 1 <= self._MAX_QT_SLIDER_VALUE:
            return bounded_value
        return int(round(
            bounded_value
            * (self._common_timestamp_count - 1)
            / self._slider_maximum
        ))

    def _common_index_to_slider_value(self, common_index):
        """把公共时间轴原始下标映射回 Qt 滑块整数位置。

        Parameters
        ----------
        common_index : Integral
            公共时间轴中的原始采样点下标。

        Returns
        -------
        int
            范围为 ``[0, self._slider_maximum]`` 的滑块位置。

        Workflow
        --------
        对十亿点以内时间轴直接返回下标；超过十亿点时等比例压缩到十亿格。
        """
        if self._common_timestamp_count <= 1 or self._slider_maximum <= 0:
            return 0
        bounded_index = min(
            max(0, int(common_index)),
            self._common_timestamp_count - 1,
        )
        if self._common_timestamp_count - 1 <= self._MAX_QT_SLIDER_VALUE:
            return bounded_index
        return int(round(
            bounded_index
            * self._slider_maximum
            / (self._common_timestamp_count - 1)
        ))

    def _nearest_common_timestamp_index(self, timestamp):
        """用懒加载二分查找离给定时间最近的公共采样点。

        Parameters
        ----------
        timestamp : Real
            需要同步到滑块的真实时间值。

        Returns
        -------
        int or None
            最近公共时间戳的原始下标；无公共时间轴时返回 None。

        Workflow
        --------
        使用与 searchsorted-left 等价的标量二分得到插入位置，再只读取插入点及
        其前一个点两个候选标量，比较绝对时间差后返回较近下标；不会读取完整
        Dataset，也不会干扰曲线时间窗口定位的调用统计。
        """
        timestamps = self.continuous_store.common_timestamps
        if timestamps is None or self._common_timestamp_count == 0:
            return None
        left = 0
        right = self._common_timestamp_count
        while left < right:
            middle = (left + right) // 2
            if timestamps[middle] < timestamp:
                left = middle + 1
            else:
                right = middle
        insertion_index = left
        if insertion_index <= 0:
            return 0
        if insertion_index >= self._common_timestamp_count:
            return self._common_timestamp_count - 1
        previous_value = float(timestamps[insertion_index - 1])
        next_value = float(timestamps[insertion_index])
        if abs(timestamp - previous_value) <= abs(next_value - timestamp):
            return insertion_index - 1
        return insertion_index

    def _update_common_timestamp_slider_label(self, slider_value=None):
        """更新滑块第三行的当前位置、总长度和时间，最多读取一个标量。

        Parameters
        ----------
        slider_value : Integral or None, optional
            要预览的滑块值；None 时使用控件当前值。

        Returns
        -------
        None
            原地更新 ``common_timestamp_slider_label``。

        Workflow
        --------
        无公共轴时显示禁用原因；否则映射为原始下标、读取该下标一个标量，并显示
        ``当前位置 / 总长度：i / N    时间：t``。时间统一使用五位有效数字，既能
        表示很大或很小的时间，也不会因为小数位数无限增长而挤压界面。
        """
        if self._common_timestamp_count == 0:
            self.common_timestamp_slider_label.setText("无公共 timestamps")
            return
        if slider_value is None:
            slider_value = self.common_timestamp_slider.value()
        common_index = self._slider_value_to_common_index(slider_value)
        timestamp = self.continuous_store.common_timestamps[common_index]
        timestamp = timestamp.item() if hasattr(timestamp, "item") else timestamp
        self.common_timestamp_slider_label.setText(
            f"当前位置 / 总长度：{common_index:,} / "
            f"{self._common_timestamp_count:,}    "
            f"时间：{float(timestamp):.5g} {self.detail_config['timestamp_unit']}"
        )

    def _preview_common_timestamp_slider_position(self, slider_value):
        """拖动滑块时预览位置，不立即重复读取和重画所有曲线。

        Parameters
        ----------
        slider_value : int
            Qt ``sliderMoved`` signal 提供的当前整数位置。

        Returns
        -------
        None
            更新预览标签；真正绘图由 120 ms 单次定时器合并执行。

        Workflow
        --------
        映射并显示当前时间点，然后重启单次定时器，使连续拖动只在停顿后重绘。
        """
        if self._syncing_time_slider:
            return
        self._update_common_timestamp_slider_label(slider_value)
        self._slider_update_timer.start()

    def _on_common_timestamp_slider_changed(self, _slider_value):
        """合并鼠标、键盘和程序之外的滑块变化请求。

        Parameters
        ----------
        _slider_value : int
            Qt ``valueChanged`` signal 的位置值；方法从控件读取最终值。

        Returns
        -------
        None
            启动 120 ms 单次定时器，避免拖动期间反复读取多个通道。

        Workflow
        --------
        程序同步滑块时直接退出；用户改变时更新标签并重启防抖定时器。
        """
        if self._syncing_time_slider:
            return
        self._update_common_timestamp_slider_label()
        self._slider_update_timer.start()

    def _apply_common_timestamp_slider_position(self):
        """把滑块当前下标对应的一个公共时间戳应用为中心时间。

        Returns
        -------
        None
            无公共时间轴时不操作；否则重载当前通道页。

        Workflow
        --------
        将滑块值映射到公共原始下标，只读取一个 timestamps 标量，将 epoch 导航
        状态改为“自由时间”，再调用 ``set_current_timestamp`` 更新窗口和图形。
        """
        common_index = self._slider_value_to_common_index(
            self.common_timestamp_slider.value()
        )
        if common_index is None:
            return
        timestamp = self.continuous_store.common_timestamps[common_index]
        timestamp = timestamp.item() if hasattr(timestamp, "item") else timestamp
        self.set_current_timestamp(
            timestamp,
            epoch_index=None,
            preserve_epoch=True,
        )

    def _effective_slider_points_per_second(self):
        """计算当前数据规模下箭头长按对应的原始采样点速度。

        Returns
        -------
        float
            每秒跨过的公共 timestamps 原始点数。时间轴不超过十亿点时等于配置
            ``detail.slider_points_per_second``；超过十亿点时按总点数比例放大。

        Workflow
        --------
        读取已验证为正数的配置速度；用 ``max(1, (N-1)/1e9)`` 计算缩放因子，
        使超大时间轴在滑块视觉坐标中的移动速度仍接近用户设置。
        """
        configured_speed = float(
            self.detail_config["slider_points_per_second"]
        )
        scale = max(
            1.0,
            max(0, self._common_timestamp_count - 1)
            / self._MAX_QT_SLIDER_VALUE,
        )
        return configured_speed * scale

    def _move_common_slider_by_original_points(self, point_delta):
        """按公共时间轴原始点数移动滑块并限制在两端。

        Parameters
        ----------
        point_delta : Integral
            带方向的原始采样点增量；负数向左，正数向右。

        Returns
        -------
        bool
            滑块实际位置发生变化时返回 True；已在边界或无公共轴时返回 False。

        Workflow
        --------
        将当前滑块值还原为公共下标，加上 point_delta 并夹在合法范围，再映射回
        Qt 滑块值。超十亿点时映射会按比例取整，不追求单点绝对精度。
        """
        if self._common_timestamp_count <= 0:
            return False
        current_index = self._slider_value_to_common_index(
            self.common_timestamp_slider.value()
        )
        target_index = min(
            max(0, current_index + int(point_delta)),
            self._common_timestamp_count - 1,
        )
        old_value = self.common_timestamp_slider.value()
        self.common_timestamp_slider.setValue(
            self._common_index_to_slider_value(target_index)
        )
        return self.common_timestamp_slider.value() != old_value

    def _start_slider_motion(self, direction):
        """开始由左/右箭头驱动的连续滑块移动。

        Parameters
        ----------
        direction : {-1, 1}
            -1 表示向左，1 表示向右。

        Returns
        -------
        None
            初始化计时状态并启动 30 ms 周期定时器。

        Workflow
        --------
        验证方向和公共轴；记录 monotonic 高精度起始时间，清空小数点余量和移动
        标记，然后启动定时器。短按若来不及触发定时器，会在释放时移动一个点。
        """
        if direction not in {-1, 1} or self._common_timestamp_count <= 0:
            return
        now = time.perf_counter()
        self._slider_motion_direction = direction
        self._slider_motion_last_time = now
        self._slider_motion_last_redraw = now
        self._slider_motion_fraction = 0.0
        self._slider_motion_moved = False
        self._slider_motion_timer.start()

    def _advance_slider_motion(self):
        """按真实经过时间推进一次长按滑块运动。

        Returns
        -------
        None
            更新滑块位置；约每 200 ms 最多重绘一次当前通道页。

        Workflow
        --------
        用 ``perf_counter`` 计算自上次 tick 的秒数，乘有效点速并累加不足一个点的
        小数余量；取得整数点数后移动。长按期间标签随滑块更新，而曲线按 200 ms
        节流重绘，兼顾可见反馈和大型 HDF5 随机读取成本。
        """
        if self._slider_motion_direction == 0:
            return
        now = time.perf_counter()
        elapsed = max(0.0, now - self._slider_motion_last_time)
        self._slider_motion_last_time = now
        point_float = (
            elapsed * self._effective_slider_points_per_second()
            + self._slider_motion_fraction
        )
        point_count = int(point_float)
        self._slider_motion_fraction = point_float - point_count
        if point_count > 0:
            moved = self._move_common_slider_by_original_points(
                self._slider_motion_direction * point_count
            )
            self._slider_motion_moved = self._slider_motion_moved or moved
        if (
                self._slider_motion_moved
                and now - self._slider_motion_last_redraw >= 0.2
        ):
            self._slider_update_timer.stop()
            self._apply_common_timestamp_slider_position()
            self._slider_motion_last_redraw = now

    def _stop_slider_motion(self):
        """停止箭头长按运动，并把最终滑块位置应用到图形。

        Returns
        -------
        None
            停止周期/防抖定时器并完成最后一次中心时间更新。

        Workflow
        --------
        若按下时间短于首个 tick，则按方向移动一个原始点；随后清除方向和计时
        状态，并立即应用最终位置，保证短按和长按都能得到确定结果。
        """
        direction = self._slider_motion_direction
        if direction == 0:
            return
        self._slider_motion_timer.stop()
        if not self._slider_motion_moved:
            self._move_common_slider_by_original_points(direction)
        self._slider_motion_direction = 0
        self._slider_motion_last_time = None
        self._slider_motion_last_redraw = None
        self._slider_motion_fraction = 0.0
        self._slider_update_timer.stop()
        self._apply_common_timestamp_slider_position()

    def apply_uniform_ylim_from_inputs(self):
        """把用户输入的纵轴范围统一应用到所有 Detail 通道子图。

        两个输入框都留空表示恢复自动纵轴。设置明确范围时会清空旧的
        ``channel_ylims``，因为“统一纵轴”应高于历史通道独立范围。

        Returns
        -------
        None
            合法时更新内存 config 并重绘；非法时显示警告且保持旧范围。

        Workflow
        --------
        同时读取最小/最大文本；均空则使用 None；否则要求两者都是有限实数且
        minimum < maximum；写入 detail.ylim、清空 channel_ylims，并重载当前页。
        """
        minimum_text = self.ylim_min_input.text().strip()
        maximum_text = self.ylim_max_input.text().strip()
        if not minimum_text and not maximum_text:
            new_ylim = None
        else:
            try:
                minimum = float(minimum_text)
                maximum = float(maximum_text)
            except ValueError:
                QMessageBox.warning(
                    self,
                    "纵轴范围错误",
                    "纵轴最小值和最大值必须同时填写有效数值；两者都留空表示自动范围。",
                )
                return
            if (
                    not np.isfinite(minimum)
                    or not np.isfinite(maximum)
                    or minimum >= maximum
            ):
                QMessageBox.warning(
                    self,
                    "纵轴范围错误",
                    "纵轴范围必须是有限数值，并满足最小值小于最大值。",
                )
                return
            new_ylim = [minimum, maximum]

        self.detail_config["ylim"] = new_ylim
        self.detail_config["channel_ylims"] = {}
        parent = self.parent()
        if isinstance(parent, DataViewer):
            parent._refresh_setting_editors()
        self.load_channel_page()

    def set_epoch_index(self, epoch_index):
        """跳转到指定原始 epoch 的 start_timestamp。

        Parameters
        ----------
        epoch_index : Integral
            HDF5 中的原始 epoch 下标，范围为 ``[0, len(epochs_store))``。

        Returns
        -------
        None
            更新当前 epoch、输入框、中心时间和图形。

        Workflow
        --------
        验证 Store、start_timestamps 和下标；只读取指定下标一个时间戳标量；保存
        current_epoch_index 和 ID；调用 set_current_timestamp，并更新导航按钮边界。
        """
        if self.epochs_store is None or self.epochs_store.start_timestamps is None:
            raise ValueError("没有可用于波形跳转的 start_timestamps。")
        normalized_index = _normalize_indices(
            epoch_index,
            len(self.epochs_store),
            "epoch_index",
        )[0]
        timestamp = self.epochs_store.start_timestamps[normalized_index]
        timestamp = timestamp.item() if hasattr(timestamp, "item") else timestamp
        self.current_epoch_index = normalized_index
        self.epoch_jump_input.setText(
            self.epochs_store.epochs_ids[normalized_index]
        )
        self.set_current_timestamp(timestamp, epoch_index=normalized_index)
        self._update_epoch_navigation_buttons()

    def jump_to_epoch_from_input(self):
        """按输入的 epoch ID 或原始整数下标执行波形跳转。

        Returns
        -------
        None
            成功时调用 ``set_epoch_index``；失败时显示中文 QMessageBox。

        Workflow
        --------
        去除输入空白；优先按完整 ID 字符串匹配，未匹配再解析整数原始下标；
        捕获不存在、格式和越界错误并提示用户。
        """
        text = self.epoch_jump_input.text().strip()
        if not text:
            QMessageBox.warning(self, "波形跳转失败", "请输入 epoch ID 或原始下标。")
            return
        try:
            if text in self._epoch_id_to_index:
                epoch_index = self._epoch_id_to_index[text]
            else:
                epoch_index = int(text)
            self.set_epoch_index(epoch_index)
        except (ValueError, IndexError) as exc:
            QMessageBox.warning(self, "波形跳转失败", str(exc))

    def _find_adjacent_epoch_from_timestamp(self, direction):
        """从自由时间位置分块查找前一个或后一个 epoch。

        Parameters
        ----------
        direction : {-1, 1}
            -1 查找不晚于当前时间的最大 start_timestamp；1 查找不早于当前时间的
            最小 start_timestamp。

        Returns
        -------
        int or None
            符合条件且时间最接近的原始 epoch 下标；不存在时返回 None。

        Workflow
        --------
        以 262,144 点为一块读取 start_timestamps；用 NumPy mask 在块内找候选；
        跨块只保留最佳时间和原始下标。不会访问任何 waveform。
        """
        if (
                self.epochs_store is None
                or self.epochs_store.start_timestamps is None
                or self.current_timestamp is None
        ):
            return None
        if direction not in {-1, 1}:
            raise ValueError("direction 只能是 -1 或 1。")

        timestamps = self.epochs_store.start_timestamps
        best_index = None
        best_timestamp = None
        chunk_size = _DEFAULT_VALIDATION_CHUNK_SIZE
        for start in range(0, len(timestamps), chunk_size):
            end = min(start + chunk_size, len(timestamps))
            chunk = np.asarray(timestamps[start:end])
            finite = np.isfinite(chunk)
            if direction < 0:
                valid_positions = np.flatnonzero(
                    finite & (chunk <= self.current_timestamp)
                )
                if len(valid_positions) == 0:
                    continue
                local_values = chunk[valid_positions]
                local_position = int(valid_positions[int(np.argmax(local_values))])
                candidate_timestamp = float(chunk[local_position])
                is_better = (
                    best_timestamp is None
                    or candidate_timestamp > best_timestamp
                )
            else:
                valid_positions = np.flatnonzero(
                    finite & (chunk >= self.current_timestamp)
                )
                if len(valid_positions) == 0:
                    continue
                local_values = chunk[valid_positions]
                local_position = int(valid_positions[int(np.argmin(local_values))])
                candidate_timestamp = float(chunk[local_position])
                is_better = (
                    best_timestamp is None
                    or candidate_timestamp < best_timestamp
                )
            if is_better:
                best_timestamp = candidate_timestamp
                best_index = start + local_position
        return best_index

    def _navigate_epoch(self, direction):
        """执行上一个或下一个波形导航的共享逻辑。

        Parameters
        ----------
        direction : {-1, 1}
            -1 表示上一个，1 表示下一个。

        Returns
        -------
        None
            成功时跳转；到达边界或无可用 epoch 时显示提示。

        Workflow
        --------
        已定位 epoch 时按原始下标加减；滑块自由时间状态时按时间分块寻找邻近
        epoch；检查边界后调用 ``set_epoch_index``。
        """
        if self.epochs_store is None or self.epochs_store.start_timestamps is None:
            QMessageBox.information(self, "波形导航", "没有可用的 start_timestamps。")
            return
        if self.current_epoch_index is None:
            target_index = self._find_adjacent_epoch_from_timestamp(direction)
        else:
            target_index = self.current_epoch_index + direction
            if target_index < 0 or target_index >= len(self.epochs_store):
                target_index = None
        if target_index is None:
            boundary_text = "上一个" if direction < 0 else "下一个"
            QMessageBox.information(self, "波形导航", f"没有{boundary_text}波形。")
            return
        self.set_epoch_index(target_index)

    def previous_epoch(self):
        """跳转到上一个原始 epoch。

        Returns
        -------
        None
            成功时更新中心时间；没有上一项时显示提示。

        Workflow
        --------
        把方向 -1 交给共享导航方法；从滑块自由时间开始时先按时间寻找最近前项。
        """
        self._navigate_epoch(-1)

    def next_epoch(self):
        """跳转到下一个原始 epoch。

        Returns
        -------
        None
            成功时更新中心时间；没有下一项时显示提示。

        Workflow
        --------
        把方向 1 交给共享导航方法；从滑块自由时间开始时先按时间寻找最近后项。
        """
        self._navigate_epoch(1)

    def _update_epoch_navigation_buttons(self):
        """根据 epoch 数据可用性和当前下标更新导航按钮。

        Returns
        -------
        None
            原地设置三个按钮和输入框的 enabled 状态。

        Workflow
        --------
        无 Store/时间戳/epoch 时全部禁用；自由时间时允许向两侧寻找；已定位 epoch
        时在首尾分别禁用上一个或下一个。
        """
        available = (
            self.epochs_store is not None
            and self.epochs_store.start_timestamps is not None
            and len(self.epochs_store) > 0
        )
        self.epoch_jump_input.setEnabled(available)
        self.jump_to_epoch_btn.setEnabled(available)
        if not available:
            self.previous_epoch_btn.setEnabled(False)
            self.next_epoch_btn.setEnabled(False)
        elif self.current_epoch_index is None:
            self.previous_epoch_btn.setEnabled(True)
            self.next_epoch_btn.setEnabled(True)
        else:
            self.previous_epoch_btn.setEnabled(self.current_epoch_index > 0)
            self.next_epoch_btn.setEnabled(
                self.current_epoch_index < len(self.epochs_store) - 1
            )

    def _read_channel_window(self, channel_name):
        """读取一个通道内所有曲线的当前真实时间窗口。

        Parameters
        ----------
        channel_name : str
            当前页中的通道真实名称。

        Returns
        -------
        dict[str, dict[str, numpy.ndarray]]
            曲线名到 ``{"timestamps": (K,), "values": (K,)}`` 的映射；不同曲线
            因时间轴不同可具有不同 K。

        Workflow
        --------
        逐曲线构造含通道、曲线和时间范围的缓存键；命中则提升为最近使用，未命中
        才访问 Store；插入后淘汰最旧条目，记录本次实际页面涉及的曲线键。
        """
        curves = {}
        for curve_name in self._enabled_curve_names(channel_name):
            cache_key = (
                channel_name,
                curve_name,
                self.start_timestamp,
                self.end_timestamp,
            )
            if cache_key in self._slice_cache:
                timestamps, values = self._slice_cache.pop(cache_key)
                self._slice_cache[cache_key] = (timestamps, values)
            else:
                timestamps, values = self.continuous_store.get_curve_slice(
                    channel_name,
                    curve_name,
                    self.start_timestamp,
                    self.end_timestamp,
                )
                cache_size = int(
                    self.config["performance"]["detail_slice_cache_size"]
                )
                if cache_size > 0:
                    self._slice_cache[cache_key] = (timestamps, values)
                    while len(self._slice_cache) > cache_size:
                        self._slice_cache.popitem(last=False)
            curves[curve_name] = {
                "timestamps": timestamps,
                "values": values,
            }
            self.last_loaded_curve_keys.append((channel_name, curve_name))
        return curves

    def apply_config(self, config):
        """应用完整新配置并重建当前 Detail 页面。

        Parameters
        ----------
        config : Mapping[str, object]
            已合并的完整配置，必须包含有效 detail/performance 区块。详见
            ``DataViewer.__init__`` 的 ``config_dict`` 参数。

        Returns
        -------
        None
            更新窗口、缓存和页面后返回。

        Workflow
        --------
        验证配置，更新时间窗口宽度、统一纵轴输入和窗口尺寸并清空旧缓存；保持
        中心时间，用新前后范围重算 start/end/xlim，再重算通道容量和加载页面。
        """
        validate_config_dict(config)
        self.config = config
        self.detail_config = config["detail"]
        self._apply_plot_visibility_from_config()
        self.time_before = float(self.detail_config["time_before"])
        self.time_after = float(self.detail_config["time_after"])
        configured_ylim = self.detail_config["ylim"]
        if configured_ylim is None:
            self.ylim_min_input.clear()
            self.ylim_max_input.clear()
        else:
            self.ylim_min_input.setText(str(configured_ylim[0]))
            self.ylim_max_input.setText(str(configured_ylim[1]))
        self.resize(
            int(self.detail_config["window_width"]),
            int(self.detail_config["window_height"]),
        )
        self._slice_cache.clear()
        if self.current_timestamp is not None:
            self.start_timestamp = self.current_timestamp - self.time_before
            self.end_timestamp = self.current_timestamp + self.time_after
            self.xlim = (self.start_timestamp, self.end_timestamp)
        self.channels_per_page = self.calculate_channels_per_page()
        self.load_channel_page()

    def load_channel_page(self):
        """只读取并绘制当前通道页，保持中心时间状态不变。

        Returns
        -------
        None
            更新 current_channel_names、Axes、页标签和按钮状态。

        Workflow
        --------
        1. 计算当前通道半开区间并清空旧 Axes。
        2. 空通道/未选 epoch 显示提示；否则逐当前页通道读取窗口并调用独立绘图函数。
        3. tight_layout、draw，并同步页码和翻页按钮；绝不遍历页外通道。
        """
        if self._plot_resources_released:
            return
        self._update_total_channel_pages()
        visible_channel_names = self._visible_channel_names()
        start_channel = self.current_channel_page * self.channels_per_page
        end_channel = min(
            start_channel + self.channels_per_page,
            len(visible_channel_names),
        )
        self.current_channel_names = visible_channel_names[start_channel:end_channel]
        self.last_loaded_curve_keys = []

        self.figure.clear()
        self.axes = []
        configured_width = float(self.detail_config["figure_width"])
        self.canvas.setFixedWidth(max(1, int(round(configured_width))))
        row_count = max(1, len(self.current_channel_names))
        canvas_height = int(self.canvas.height())
        configured_height = row_count * float(self.detail_config["subplot_height"])
        # Figure 与 Canvas 始终采用配置的像素宽度；这样首次打开和滑块重绘后的
        # 子图宽度一致，同时不会把用户设置的 detail.figure_width 覆盖掉。
        figure_width = configured_width
        figure_height = canvas_height if canvas_height > 1 else configured_height
        self.figure.set_size_inches(
            max(1.0, figure_width / self.figure.dpi),
            max(1.0, figure_height / self.figure.dpi),
            forward=False,
        )

        if not self.current_channel_names:
            axis = self.figure.add_subplot(111)
            empty_message = (
                "未勾选任何 Detail 子图"
                if self.channel_names
                else "continuous 数据中没有通道"
            )
            axis.text(
                0.5,
                0.5,
                empty_message,
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.axes.append(axis)
        elif self.current_timestamp is None:
            axis = self.figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "请双击 Overview 中的 epoch 以加载连续数据",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.axes.append(axis)
        else:
            target_width = (
                max(1, int(round(float(self.detail_config["figure_width"]))))
            )
            for row_index, channel_name in enumerate(
                    self.current_channel_names,
                    start=1,
            ):
                axis = self.figure.add_subplot(row_count, 1, row_index)
                enabled_curve_names = self._enabled_curve_names(channel_name)
                if not enabled_curve_names:
                    axis.text(
                        0.5,
                        0.5,
                        "未勾选任何曲线",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        color="#b45309",
                    )
                    axis.set_title(channel_name, loc="left", fontsize=11, fontweight="bold")
                    axis.set_xlim(self.start_timestamp, self.end_timestamp)
                    axis.set_ylabel("幅值")
                    axis.grid(True, alpha=0.25, linestyle="--")
                else:
                    curves = self._read_channel_window(channel_name)
                    plot_detail_channel(
                        axis,
                        channel_name,
                        curves,
                        self.epoch_center_timestamp,
                        self.start_timestamp,
                        self.end_timestamp,
                        self.config,
                        target_width=target_width,
                        epoch_region_start=self.epoch_region_start,
                        epoch_region_end=self.epoch_region_end,
                    )
                if row_index == row_count:
                    axis.set_xlabel(
                        f"时间 ({self.detail_config['timestamp_unit']})"
                    )
                self.axes.append(axis)

        self.figure.tight_layout()
        # 统一绘图区左右边界，避免不同时间窗内的图例/标题尺寸让 Axes 宽度
        # 在重绘时变化；right 接近 1 使子图始终延伸到 Canvas 右边缘。
        self.figure.subplots_adjust(
            left=0.07,
            right=0.995,
            top=0.96,
            bottom=0.12,
            hspace=0.35,
        )
        self.canvas.draw()
        self.channel_page_label.setText(
            f"第 {self.current_channel_page + 1} / "
            f"{self.total_channel_pages} 页 | "
            f"每页 {self.channels_per_page} 个通道"
        )
        self.previous_channel_page_btn.setEnabled(self.current_channel_page > 0)
        self.next_channel_page_btn.setEnabled(
            self.current_channel_page < self.total_channel_pages - 1
        )

    def previous_channel_page(self):
        """切换到上一通道页。

        Returns
        -------
        None
            已在第一页时不执行操作。

        Workflow
        --------
        页码大于 0 时减 1 并重载；不修改 current/start/end timestamp 或 xlim。
        """
        if self.current_channel_page > 0:
            self.current_channel_page -= 1
            self.load_channel_page()

    def next_channel_page(self):
        """切换到下一通道页。

        Returns
        -------
        None
            已在末页时不执行操作。

        Workflow
        --------
        未到末页时页码加 1 并重载；不修改任何时间状态。
        """
        if self.current_channel_page < self.total_channel_pages - 1:
            self.current_channel_page += 1
            self.load_channel_page()

    def resizeEvent(self, event):
        """窗口高度变化时重算每页容量，并保留当前通道锚点。

        Parameters
        ----------
        event : PyQt5.QtGui.QResizeEvent
            Qt 传入的新旧窗口尺寸事件。

        Returns
        -------
        None
            必要时重新加载通道页。

        Workflow
        --------
        先执行父类逻辑；UI 尚未就绪或已释放时退出；容量变化时根据原页首通道
        的全局位置计算新页码并重载。
        """
        super().resizeEvent(event)
        if not self._ui_ready or self._plot_resources_released:
            return

        old_channels_per_page = self.channels_per_page
        new_channels_per_page = self.calculate_channels_per_page()
        if new_channels_per_page == old_channels_per_page:
            return

        first_channel_index = self.current_channel_page * old_channels_per_page
        self.channels_per_page = new_channels_per_page
        self.current_channel_page = first_channel_index // self.channels_per_page
        self.load_channel_page()

    def release_plot_resources(self):
        """幂等释放 Detail 绘图对象和切片缓存。

        Returns
        -------
        None
            完成释放或此前已释放时返回。

        Workflow
        --------
        设置释放标记；清空 Figure，关闭并脱离 Canvas/toolbar；清空 Axes、当前
        通道、读取记录和 LRU 缓存；最后将重资源引用设为 None。
        """
        if self._plot_resources_released:
            return
        self._plot_resources_released = True
        self._slider_update_timer.stop()
        self._slider_motion_timer.stop()
        self._slider_motion_direction = 0
        if self._curve_settings_dialog is not None:
            self._curve_settings_dialog.close()
            self._curve_settings_dialog = None
        old_figure = self.figure
        old_canvas = self.canvas
        if old_figure is not None:
            old_figure.clear()
        if old_canvas is not None:
            old_canvas.close()
            old_canvas.setParent(None)
        if self.toolbar is not None:
            self.toolbar.close()
        self.axes = []
        self.current_channel_names = []
        self.last_loaded_curve_keys = []
        self._slice_cache.clear()
        self.figure = None
        self.canvas = None
        self.toolbar = None

class DataViewer(QMainWindow):
    """从 working folder 管理配置、HDF5 生命周期和全部 App 窗口。

    主窗口只保存原始 epoch 下标、布尔 properties 和当前页 Widget。公开构造器
    拥有它打开的 Store，并在真正关闭窗口时统一释放；不会常驻全部 waveform。
    """

    overview_double_clicked = pyqtSignal(int, object)

    _SETTING_FIELDS = (
        ("overview", "rows_per_page", "Overview 每页行数", "int"),
        ("overview", "cols_per_page", "Overview 每页列数", "int"),
        ("overview", "window_width", "主窗口宽度", "int"),
        ("overview", "window_height", "主窗口高度", "int"),
        ("overview", "subplot_min_height", "Overview 子图最小高度", "float"),
        ("overview", "xlim", "Overview xlim", "range"),
        ("overview", "ylim", "Overview ylim", "range"),
        ("overview", "line_color", "Overview 线条颜色", "str"),
        ("overview", "line_width", "Overview 线宽", "float"),
        ("overview", "figure_dpi", "Overview DPI", "float"),
        ("detail", "subplot_height", "Detail 子图高度", "float"),
        ("detail", "figure_width", "Detail Figure 宽度", "float"),
        ("detail", "window_width", "Detail 窗口宽度", "int"),
        ("detail", "window_height", "Detail 窗口高度", "int"),
        ("detail", "time_before", "Detail 向前时间", "float"),
        ("detail", "time_after", "Detail 向后时间", "float"),
        ("detail", "timestamp_unit", "时间单位", "str"),
        ("detail", "ylim", "Detail 默认纵轴范围", "range"),
        ("detail", "channel_ylims", "各通道 ylim（JSON）", "json"),
        ("detail", "line_width", "Detail 线宽", "float"),
        ("detail", "show_legend", "显示图例", "bool"),
        ("detail", "max_points_per_pixel", "每像素最大点数", "float"),
        ("detail", "slider_points_per_second", "滑块长按速度（点/秒）", "float"),
        ("filters", "default_state", "新属性默认筛选", "filter"),
        ("storage", "overwrite_existing", "允许覆盖已有文件", "bool"),
        ("storage", "waveforms_compression", "waveform 压缩", "nullable_str"),
        ("storage", "waveforms_compression_level", "waveform 压缩级别", "nullable_int"),
        ("storage", "waveforms_chunk_points", "waveform chunk 点数", "int"),
        ("storage", "continuous_compression", "continuous 压缩", "nullable_str"),
        ("storage", "continuous_compression_level", "continuous 压缩级别", "nullable_int"),
        ("storage", "continuous_chunk_points", "continuous chunk 点数", "int"),
        ("storage", "properties_chunk_size", "properties chunk 大小", "int"),
        ("performance", "detail_slice_cache_size", "Detail 切片缓存数", "int"),
        ("performance", "show_import_progress", "显示导入进度", "bool"),
    )

    def __init__(
            self,
            working_folder,
            epochs_dict=None,
            continuous_dict=None,
            config_dict=None,
            overwrite=None,
    ):
        """按固定文件规则创建或加载 App。

        Parameters
        ----------
        working_folder : str or Path
            App 工作目录。目录不存在时会自动创建；目录内固定使用
            ``epochs.h5``、``continuous.h5`` 和 ``config.json`` 三个文件。

        epochs_dict : Mapping[str, object] or None, optional
            首次导入或显式覆盖 epoch 数据时使用的字典。设 epoch 总数为 ``E``，
            支持以下结构::

                {
                    "waveforms": waveforms,
                    "start_timestamps": start_timestamps,  # 可省略
                    "epochs_ids": epochs_ids,            # 可省略
                    "properties": {                        # 可省略
                        "is_delete": is_delete,             # 可省略
                        "其他属性名": bool_values,
                    },
                }

            ``waveforms`` 是唯一必需字段，表示 E 条独立的 epoch 波形。它可以是：

            - 长度为 E 的 list/tuple；第 ``i`` 项是形状 ``(Ni,)`` 的一维数值
              list 或 ndarray。不同 epoch 的 ``Ni`` 可以不同；
            - dtype=object、形状 ``(E,)`` 的 ndarray；每个元素是一条形状
              ``(Ni,)`` 的一维数值数组；
            - 形状 ``(E, N)`` 的普通二维数值 ndarray；此时第一维是 epoch，
              第二维是该 epoch 内的采样点，所有波形恰好等长；
            - 其他支持 ``len(waveforms)`` 和 ``waveforms[i]`` 的懒数据源，且
              ``waveforms[i]`` 必须支持一维切片。

            每条 waveform 的唯一维度都表示该 epoch 内按采样顺序排列的波形点，
            元素必须是数值，不能是二维矩阵或布尔数据。波形横坐标默认使用
            ``0 .. Ni-1`` 的采样点位置。

            ``start_timestamps`` 可为长度 E 的一维数值 list/ndarray/懒数据源，
            形状为 ``(E,)``。第 i 个值是第 i 条 epoch 在 continuous 时间坐标中的
            真实开始时间，不是数组下标；时间单位由配置的 ``timestamp_unit``
            说明。省略或传 None 时不会创建 HDF5 Dataset，Overview 仍可使用，
            但双击 epoch 时会提示无法定位 Detail。

            ``epochs_ids`` 可为长度 E 的一维字符串序列，形状为 ``(E,)``；ID
            必须唯一。省略或传 None 时自动生成 ``epoch_0`` 到 ``epoch_{E-1}``。

            ``properties`` 可省略，也可为“属性名 -> 一维 bool 数据”的 Mapping。
            每个 bool list/ndarray 的形状必须为 ``(E,)``，第 i 个布尔值描述第 i
            条原始 epoch。省略 properties 时自动创建全 False 的 ``is_delete``；
            传了其他属性但缺少 ``is_delete`` 时也会自动补齐。属性只允许一层，
            不应再嵌套第二个 ``properties``。

            传入该字典时，数据通过验证后原子写入 ``epochs.h5``。未传入时加载
            已有文件；字典和文件都不存在会抛出 ValueError。文件已存在时默认
            拒绝覆盖，只有有效配置的 ``storage.overwrite_existing=True`` 才覆盖。

        continuous_dict : Mapping[str, object] or None, optional
            首次导入或显式覆盖连续数据时使用的动态嵌套字典。通道数和每个通道
            的曲线数都没有固定上限，支持以下结构::

                {
                    "__common_timestamps__": common_timestamps,  # 可省略
                    "通道名称1": {
                        "曲线名称1": {
                            "values": values,
                            "timestamps": timestamps,            # 可省略/None
                        },
                        "曲线名称2": {
                            "values": values,
                        },
                    },
                    "通道名称2": {
                        "曲线名称": {
                            "values": values,
                            "timestamps": own_timestamps,
                        },
                    },
                }

            ``__common_timestamps__`` 是保留键，不是通道。它可为形状 ``(Nc,)``
            的一维实数 list/ndarray/memmap/h5py Dataset/懒数据源；唯一维度表示
            连续采样点，数值必须有限且单调非递减。使用公共时间轴的曲线不会在
            HDF5 中重复保存 timestamps。

            其余顶层键是任意非空通道名称。每个通道值是“曲线名 -> 曲线数据”
            的 Mapping。同一通道中的多条曲线绘制在同一个 Axes，不同通道各占
            一个子图。通道名和曲线名可含中文、空格等字符，原名以 UTF-8
            attribute 保存，内部 HDF5 路径使用安全编号。

            每条曲线的 ``values`` 必需，支持一维数值 list/ndarray/memmap/
            h5py Dataset/懒数据源，形状为 ``(M,)``；唯一维度表示按时间顺序的 M
            个连续采样值。曲线自己的 ``timestamps`` 若存在，也必须为形状
            ``(M,)`` 的一维有限实数数据，且单调非递减，与 values 严格等长。

            ``timestamps`` 缺失或为 None 时自动使用公共时间轴，此时 M 必须等于
            Nc。若曲线既没有自身 timestamps，也没有公共时间轴，会立即报错。
            不同曲线允许不同的 M 和不同时间轴，但它们必须与 epoch 的
            start_timestamps 使用同一时间坐标及单位。

            未传 continuous_dict 时优先加载已有 ``continuous.h5``；字典和文件
            都不存在仍允许启动 Overview，但双击时会提示没有 continuous 数据。
            传入字典且文件已存在时，同样由 ``overwrite_existing`` 决定是否覆盖。

        config_dict : Mapping[str, object] or None, optional
            用户只需传想覆盖的配置字段，例如
            ``{"overview": {"rows_per_page": 4}}``。完整默认结构由模块级
            ``DEFAULT_CONFIG`` 定义，并通过 ``merge_config_dict()`` 递归合并；
            各字段类型和范围由 ``validate_config_dict()`` 检查，因此此处不重复
            展开全部配置键。

            未传入时：若工作目录存在 ``config.json`` 则自动加载、补齐和验证；
            否则使用 ``merge_config_dict(None)`` 得到默认配置。传入时：若 JSON
            不存在则保存合并结果；若已存在则必须在本次 config_dict 中明确传入
            ``{"storage": {"overwrite_existing": True}}`` 才允许覆盖。UI 中点击
            “保存设置”属于用户显式保存，不受首次调用的文件冲突规则限制。
        
        overwrite: bool or None
            写入时是否覆盖已有的文件，为None时跟随config文件的设置。

        Raises
        ------
        ValueError or FileExistsError
            working folder、输入数据或文件冲突不符合规范时抛出。

        Returns
        -------
        None
            构造函数不返回值；成功后实例拥有已打开 Store 和完整 Overview UI。

        Workflow
        --------
        1. 验证/创建工作目录，按 config.json 规则确定 effective config。
        2. 对 epochs.h5、continuous.h5 分别执行“加载、创建、覆盖或允许缺失”规则。
        3. 任一步失败先关闭已打开句柄；成功后把 Store 交给 `_initialize_viewer`。
        """
        super().__init__()
        self.working_folder = _resolve_working_folder(working_folder)
        self.config_path = self.working_folder / "config.json"
        effective_config = _resolve_config_for_working_folder(
            self.working_folder,
            config_dict,
        )
        if overwrite is None:
            overwrite = bool(effective_config["storage"]["overwrite_existing"])
        epochs_path = self.working_folder / "epochs.h5"
        continuous_path = self.working_folder / "continuous.h5"

        epochs_store = None
        continuous_store = None
        try:
            if epochs_dict is None:
                if not epochs_path.exists():
                    raise ValueError(
                        "未传入 epochs_dict，working_folder 中也不存在 epochs.h5。"
                    )
                epochs_store = EpochsHDF5Store(epochs_path, mode="r+")
            else:
                epochs_store = EpochsHDF5Store.create(
                    epochs_path,
                    epochs_dict,
                    effective_config,
                    overwrite=overwrite,
                )

            if continuous_dict is None:
                if continuous_path.exists():
                    continuous_store = ContinuousHDF5Store(continuous_path, mode="r")
            else:
                continuous_store = ContinuousHDF5Store.create(
                    continuous_path,
                    continuous_dict,
                    effective_config,
                    overwrite=overwrite,
                )
        except Exception:
            if continuous_store is not None:
                continuous_store.close()
            if epochs_store is not None:
                epochs_store.close()
            raise

        self._owns_stores = True
        self._initialize_viewer(epochs_store, continuous_store, effective_config)

    @classmethod
    def _from_open_stores_for_testing(
            cls,
            epochs_store,
            config_dict=None,
            continuous_store=None,
    ):
        """让回归测试注入已打开且可打探针的 Store。

        Parameters
        ----------
        epochs_store : EpochsHDF5Store
            测试持有的已打开 epoch Store。
        config_dict : Mapping[str, object] or None, optional
            测试配置。支持结构详见 ``DataViewer.__init__`` 的 ``config_dict`` 参数。
        continuous_store : ContinuousHDF5Store or None, optional
            测试持有的 continuous Store。

        Returns
        -------
        DataViewer
            不拥有两个 Store 生命周期的窗口；调用方测试负责关闭 Store。

        Workflow
        --------
        绕过公开路径解析，仅验证 Store 和配置，创建 QMainWindow 实例并调用统一
        初始化方法。此方法带下划线，不是外部业务接口。
        """
        instance = cls.__new__(cls)
        QMainWindow.__init__(instance)
        if not isinstance(epochs_store, EpochsHDF5Store):
            raise ValueError("epochs_store 必须是 EpochsHDF5Store。")
        epochs_store._ensure_open()
        if continuous_store is not None:
            if not isinstance(continuous_store, ContinuousHDF5Store):
                raise ValueError("continuous_store 必须是 ContinuousHDF5Store 或 None。")
            continuous_store._ensure_open()
        effective_config = merge_config_dict(config_dict)
        validate_config_dict(effective_config)
        instance.working_folder = Path(epochs_store.file_path).parent
        instance.config_path = instance.working_folder / "config.json"
        instance._owns_stores = False
        instance._initialize_viewer(
            epochs_store,
            continuous_store,
            effective_config,
        )
        return instance

    def _initialize_viewer(self, epochs_store, continuous_store, effective_config):
        """初始化与文件解析无关的运行状态和 UI。

        Parameters
        ----------
        epochs_store : EpochsHDF5Store
            已打开的 epoch 数据源。
        continuous_store : ContinuousHDF5Store or None
            已打开的连续数据源；None 表示 Detail 不可用。
        effective_config : dict[str, object]
            与默认值合并并验证过的完整配置。原始 config_dict 的规则详见
            ``DataViewer.__init__`` 的同名参数。

        Returns
        -------
        None
            完成状态、筛选、分页和 UI 初始化。

        Workflow
        --------
        缓存配置与页容量；取得小型 properties 原数组；恢复筛选状态；建立可见原始
        下标和 dirty/选择容器；搭建 UI 并只加载第一页 waveform。
        """
        self.epochs_store = epochs_store
        self.continuous_store = continuous_store
        self.config = effective_config
        self.overview_config = self.config["overview"]

        self.rows_per_page = self.overview_config["rows_per_page"]
        self.cols_per_page = self.overview_config["cols_per_page"]
        self.items_per_page = self.rows_per_page * self.cols_per_page

        # properties 相对 waveform 很小，Store 已将它们作为内存镜像管理。
        # 筛选始终使用这些布尔数组，不会触及 waveform Dataset。
        self.properties = self.epochs_store.get_properties(copy=False)
        filter_config = self.config["filters"]
        self.filter_states = {
            property_name: filter_config["property_states"].get(
                property_name,
                filter_config["default_state"],
            )
            for property_name in self.properties
        }
        self.property_filter_combos = {}
        self.dirty_property_indices = {
            property_name: set() for property_name in self.properties
        }
        # 保存每个 dirty 下标在首次修改前的值，用于“放弃修改”。
        self._dirty_original_values = {
            property_name: {} for property_name in self.properties
        }

        initial_mask = build_property_filter_mask(
            self.properties,
            self.filter_states,
        )
        # 只保存稳定的原始下标，不保存筛选后的 waveform 副本。
        self.visible_epoch_indices = np.flatnonzero(initial_mask).astype(
            np.int64,
            copy=False,
        )
        self.page_epoch_indices = np.empty(0, dtype=np.int64)
        self.selected_epoch_indices = set()
        self.current_overview_widgets = []
        self.last_clicked_epoch_index = None
        self.last_double_click_info = None
        # Detail 采用懒创建：首次成功双击时创建，之后始终复用同一实例。
        self.detail_window = None
        self._last_detail_warning_box = None
        self.current_page = 0
        self.total_pages = 1
        self._is_closing = False
        self._stores_closed = False
        self._delete_working_folder_on_close = False
        self._working_folder_deleted = False
        self.setting_editors = {}

        self.calculate_total_pages()
        self.init_ui()
        self.load_page()

    def calculate_total_pages(self):
        """根据可见原始下标数量计算 Overview 总页数。

        Returns
        -------
        None
            更新 ``total_pages`` 并把 ``current_page`` 夹在合法范围。

        Workflow
        --------
        用可见下标数对 items_per_page 向上取整；空结果仍保留一页，不读 waveform。
        """
        visible_count = len(self.visible_epoch_indices)
        self.total_pages = max(
            1,
            (visible_count + self.items_per_page - 1) // self.items_per_page,
        )
        self.current_page = min(max(0, self.current_page), self.total_pages - 1)

    def init_ui(self):
        """建立主窗口 Overview、设置、动态筛选和操作控件。

        Returns
        -------
        None
            将所有 Qt 控件和布局保存为实例属性并连接 signals。

        Workflow
        --------
        设置窗口样式；建立状态区和默认折叠的全配置编辑区；按 property 动态创建
        All/True/False 控件；建立 waveform 网格、属性/保存及翻页按钮；最后同步状态。
        """
        self.setWindowTitle("Epoch Overview - 单击选择 | Shift 连续多选")
        self.resize(
            self.overview_config["window_width"],
            self.overview_config["window_height"],
        )
        self.setStyleSheet(APP_STYLE_SHEET)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(0)

        # 顶部三栏比例为 1/2、1/3、1/6。波形信息栏内部再按 1/4、1/8、1/8
        # 的总窗口比例分配 ID、原始下标和开始时间，确保长下标仍有稳定空间。
        waveform_info_widget = QWidget()
        waveform_info_widget.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Fixed,
        )
        waveform_info_layout = QHBoxLayout(waveform_info_widget)
        waveform_info_layout.setContentsMargins(0, 0, 0, 0)
        waveform_info_layout.setSpacing(2)
        self.waveform_id_label = QLabel("波形ID：-")
        self.waveform_index_label = QLabel("index：-")
        self.waveform_time_label = QLabel("时间：-")
        for label in (
                self.waveform_id_label,
                self.waveform_index_label,
                self.waveform_time_label,
        ):
            label.setObjectName("infoLabel")
            label.setWordWrap(False)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        waveform_info_layout.addWidget(self.waveform_id_label, 2)
        waveform_info_layout.addWidget(self.waveform_index_label, 1)
        waveform_info_layout.addWidget(self.waveform_time_label, 1)

        # 内部状态标签供状态统计使用，不占用可见布局空间。
        self.info_label = QLabel()
        self.info_label.setVisible(False)
        self.hint_label = QLabel(
            "点击选中或取消，按住shift连续选择，双击打开detai窗口"
        )
        self.hint_label.setObjectName("infoLabel")
        self.hint_label.setStyleSheet(
            "color: #0f766e; font-size: 12px; font-weight: 500;"
        )
        self.hint_label.setWordWrap(False)
        self.hint_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        actions_widget = QWidget()
        actions_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(6)
        self.selection_label = QLabel("已选择：0")
        self.selection_label.setObjectName("infoLabel")
        self.selection_label.setStyleSheet(
            "font-weight: 600; color: #1e3a8a; background-color: #dbeafe;"
            "padding: 3px 10px; border-radius: 12px;"
        )
        self.selection_label.setWordWrap(False)
        self.selection_label.setAlignment(Qt.AlignCenter)
        self.selection_label.setMinimumWidth(64)
        self.selection_label.setFixedHeight(30)
        self.selection_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.settings_toggle_btn = QPushButton("设置")
        self.settings_toggle_btn.setObjectName("settingsToggleButton")
        self.settings_toggle_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        actions_layout.addWidget(self.selection_label)
        actions_layout.addWidget(self.settings_toggle_btn)

        info_layout.addWidget(waveform_info_widget, 3)
        info_layout.addWidget(self.hint_label, 2)
        info_layout.addWidget(actions_widget, 1)
        main_layout.addLayout(info_layout)

        # 双击记录用于程序状态追踪，不占用 Overview 的可见垂直空间。
        self.double_click_label = QLabel()
        self.double_click_label.setVisible(False)

        self.settings_window = QDialog(self)
        self.settings_window.setWindowTitle("App 设置")
        self.settings_window.setStyleSheet(APP_STYLE_SHEET)
        self.settings_window.resize(900, 520)
        settings_dialog_layout = QVBoxLayout(self.settings_window)
        settings_group = QGroupBox("App 设置（修改后点击“保存设置”）", self.settings_window)
        self.settings_group = settings_group
        settings_group.setObjectName("settingsGroup")
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setMinimumHeight(175)
        settings_content = QWidget()
        settings_grid = QGridLayout(settings_content)
        settings_grid.setContentsMargins(8, 8, 8, 8)
        settings_grid.setHorizontalSpacing(8)
        settings_grid.setVerticalSpacing(5)
        for field_index, (section, key, label_text, value_type) in enumerate(
                self._SETTING_FIELDS
        ):
            row = field_index // 3
            field_column = (field_index % 3) * 2
            settings_grid.addWidget(QLabel(f"{label_text}:"), row, field_column)
            current_value = self.config[section][key]
            if value_type in {"bool", "filter"}:
                editor = QComboBox()
                if value_type == "bool":
                    editor.addItem("True", True)
                    editor.addItem("False", False)
                else:
                    editor.addItem("All", "all")
                    editor.addItem("True", "true")
                    editor.addItem("False", "false")
                editor.setCurrentIndex(editor.findData(current_value))
            else:
                editor = QLineEdit(self._format_setting_value(current_value))
                editor.setMinimumWidth(115)
            editor.setObjectName(f"setting_{section}_{key}")
            editor.setProperty("settingValueType", value_type)
            self.setting_editors[(section, key)] = editor
            settings_grid.addWidget(editor, row, field_column + 1)
        settings_scroll.setWidget(settings_content)
        group_layout = QVBoxLayout(settings_group)
        group_layout.addWidget(settings_scroll)
        self.save_settings_btn = QPushButton("保存设置")
        self.save_settings_btn.setObjectName("saveSettingsButton")
        self.save_settings_btn.clicked.connect(
            lambda _checked=False: self.save_settings(show_message=True)
        )
        group_layout.addWidget(self.save_settings_btn, alignment=Qt.AlignRight)
        settings_dialog_layout.addWidget(settings_group)
        self.settings_toggle_btn.clicked.connect(self._show_settings_window)

        # property 数量是动态的，水平滚动区可避免属性较多时撑大主窗口。
        filter_content = QWidget()
        filter_layout = QHBoxLayout(filter_content)
        filter_layout.setContentsMargins(8, 6, 8, 6)
        filter_layout.addWidget(QLabel("属性筛选（条件之间为 AND）："))
        state_labels = (("All", "all"), ("True", "true"), ("False", "false"))
        for property_name in self.properties:
            filter_layout.addWidget(QLabel(f"{property_name}:"))
            combo = QComboBox()
            combo.setObjectName(f"propertyFilter_{property_name}")
            for label, state in state_labels:
                combo.addItem(label, state)
            combo.setCurrentIndex(combo.findData(self.filter_states[property_name]))
            combo.currentIndexChanged.connect(
                lambda _index, name=property_name: self._on_property_filter_changed(name)
            )
            self.property_filter_combos[property_name] = combo
            filter_layout.addWidget(combo)
        filter_layout.addStretch()
        filter_scroll_area = QScrollArea()
        filter_scroll_area.setWidgetResizable(True)
        filter_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        filter_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        filter_scroll_area.setFrameShape(QFrame.StyledPanel)
        filter_scroll_area.setFixedHeight(58)
        filter_scroll_area.setWidget(filter_content)
        main_layout.addWidget(filter_scroll_area)

        property_actions = QHBoxLayout()
        property_actions.addWidget(QLabel("属性："))
        self.add_property_name_combo = QComboBox()
        self.add_property_name_combo.setObjectName("addPropertyNameCombo")
        self.add_property_name_combo.setMinimumWidth(150)
        self.add_property_name_combo.addItem("请选择属性", None)
        for property_name in self.properties:
            self.add_property_name_combo.addItem(property_name, property_name)
        # 这两个别名让外部自动化测试和扩展代码可以使用更短的控件名称。
        self.property_name_combo = self.add_property_name_combo
        property_actions.addWidget(self.add_property_name_combo)

        self.add_property_value_combo = QComboBox()
        self.add_property_value_combo.setObjectName("addPropertyValueCombo")
        self.add_property_value_combo.setMinimumWidth(90)
        self.add_property_value_combo.addItem("请选择值", None)
        self.add_property_value_combo.addItem("True", True)
        self.add_property_value_combo.addItem("False", False)
        self.property_value_combo = self.add_property_value_combo
        property_actions.addWidget(self.add_property_value_combo)

        self.add_property_btn = QPushButton("添加属性")
        self.add_property_btn.setObjectName("addPropertyButton")
        self.add_property_btn.setStyleSheet(
            "background-color: #2563eb; border-color: #2563eb;"
            "padding: 7px 14px;"
        )
        self.save_properties_btn = QPushButton("保存属性")
        self.dirty_label = QLabel("未保存：0")
        property_actions.addStretch()
        property_actions.addWidget(self.dirty_label)
        property_actions.addWidget(self.add_property_btn)
        property_actions.addWidget(self.save_properties_btn)
        main_layout.addLayout(property_actions)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(
            "QScrollArea { border: none; background-color: transparent; }"
        )
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(15)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        scroll_area.setWidget(self.grid_widget)
        main_layout.addWidget(scroll_area, 1)

        controls = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 上一页")
        self.next_btn = QPushButton("下一页 ▶")
        self.page_label = QLabel()
        self.page_label.setStyleSheet(
            "font-weight: bold; font-size: 14px; padding: 0 15px;"
        )
        self.clear_page_btn = QPushButton("清除本页选中")
        self.clear_all_btn = QPushButton("清除全部选中")
        # 两个按钮位于窗口最底部，使用紧凑固定宽度可避免新增文字把小窗口
        # 的整体最小宽度强制放大；文字仍完整显示。
        self.clear_page_btn.setFixedWidth(99)
        self.clear_all_btn.setFixedWidth(99)

        controls.addWidget(self.prev_btn)
        controls.addWidget(self.page_label)
        controls.addWidget(self.next_btn)
        controls.addStretch()
        controls.addWidget(self.clear_page_btn)
        controls.addWidget(self.clear_all_btn)
        main_layout.addLayout(controls)

        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn.clicked.connect(self.next_page)
        self.clear_page_btn.clicked.connect(self.clear_current_page_selections)
        self.clear_all_btn.clicked.connect(self.clear_all_selections)
        self.add_property_name_combo.currentIndexChanged.connect(
            lambda _index=0: self._update_add_property_button_state()
        )
        self.add_property_value_combo.currentIndexChanged.connect(
            lambda _index=0: self._update_add_property_button_state()
        )
        self.add_property_btn.clicked.connect(
            lambda _checked=False: self.add_property_to_selected_epochs()
        )
        self.save_properties_btn.clicked.connect(
            lambda _checked=False: self.save_dirty_properties(show_message=True)
        )
        self.add_property_btn.setEnabled(False)
        self.update_status_labels()

    def _show_settings_window(self):
        """显示独立的设置窗口并把焦点放到该窗口。

        Returns
        -------
        None
            设置窗口以非模态方式显示，不阻塞 Overview 的选择和分页操作。

        Workflow
        --------
        调整设置窗口到可见状态，提升到顶层并激活；窗口关闭后设置仍保留在主窗口
        的编辑器中，用户可再次点击“设置”继续修改。
        """
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    @staticmethod
    def _format_setting_value(value):
        """把单个设置值转换为适合 QLineEdit 的文本。

        Parameters
        ----------
        value : object
            None、标量、Mapping、list、tuple 或 ndarray 配置值。

        Returns
        -------
        str
            None 为 ``"null"``；容器为紧凑 UTF-8 JSON；其他值为 ``str(value)``。

        Workflow
        --------
        按 None、容器、普通标量三类格式化；容器先递归转换 NumPy 类型。
        """
        if value is None:
            return "null"
        if isinstance(value, (Mapping, list, tuple, np.ndarray)):
            return json.dumps(
                convert_to_json_compatible(value),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return str(value)

    @staticmethod
    def _parse_setting_text(text, value_type, field_name):
        """按字段类型解析一个设置输入框的文本。

        Parameters
        ----------
        text : str
            QLineEdit 当前文本。
        value_type : {"int", "float", "nullable_int", "nullable_str", "range", "json", "str"}
            `_SETTING_FIELDS` 中定义的解析策略。
        field_name : str
            面向用户的中文字段名，用于错误提示。

        Returns
        -------
        int, float, str, None, list or dict
            与解析策略对应的 Python/JSON 原生值；range 返回 None 或 list。

        Workflow
        --------
        去除空白，按策略转换；nullable 接受空/none/null；range/json 使用 json.loads；
        捕获解析异常并补充字段名后重新抛出 ValueError。
        """
        stripped = text.strip()
        try:
            if value_type == "int":
                return int(stripped)
            if value_type == "float":
                return float(stripped)
            if value_type == "nullable_int":
                return None if stripped.lower() in {"", "none", "null"} else int(stripped)
            if value_type == "nullable_str":
                return None if stripped.lower() in {"", "none", "null"} else stripped
            if value_type == "range":
                if stripped.lower() in {"", "none", "null"}:
                    return None
                value = json.loads(stripped)
                if not isinstance(value, list):
                    raise ValueError("范围必须写成 JSON 列表")
                return value
            if value_type == "json":
                return json.loads(stripped)
            return stripped
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"设置“{field_name}”格式错误：{exc}") from exc

    def collect_settings_from_ui(self):
        """读取全部设置控件和动态 property 筛选。

        Returns
        -------
        dict[str, object]
            深复制、包含当前 UI 值且通过完整验证的新配置。

        Workflow
        --------
        从当前 config 深复制；逐 `_SETTING_FIELDS` 读取 ComboBox 或解析文本；若窗口
        尺寸字段未手工改写则记录实际尺寸；写入动态 filter_states；最后完整验证。

        Raises
        ------
        ValueError
            任一输入格式或配置语义无效时抛出，不修改当前生效配置。
        """
        candidate = deepcopy(self.config)
        labels = {
            (section, key): label
            for section, key, label, _value_type in self._SETTING_FIELDS
        }
        for section, key, _label, value_type in self._SETTING_FIELDS:
            editor = self.setting_editors[(section, key)]
            if value_type in {"bool", "filter"}:
                value = editor.currentData()
            else:
                value = self._parse_setting_text(
                    editor.text(),
                    value_type,
                    labels[(section, key)],
                )
            candidate[section][key] = value

        # 用户只通过拖动窗口调整大小时，也应能用“保存设置”持久化实际尺寸。
        for key, actual_size in (
                ("window_width", self.width()),
                ("window_height", self.height()),
        ):
            editor = self.setting_editors[("overview", key)]
            old_text = self._format_setting_value(self.config["overview"][key])
            if editor.text().strip() == old_text:
                candidate["overview"][key] = int(actual_size)
        if self.detail_window is not None:
            for key, actual_size in (
                    ("window_width", self.detail_window.width()),
                    ("window_height", self.detail_window.height()),
            ):
                editor = self.setting_editors[("detail", key)]
                old_text = self._format_setting_value(self.config["detail"][key])
                if editor.text().strip() == old_text:
                    candidate["detail"][key] = int(actual_size)

        candidate["filters"]["property_states"] = dict(self.filter_states)
        validate_config_dict(candidate)
        return candidate

    def _refresh_setting_editors(self):
        """让全部设置控件与当前已保存配置保持一致。

        Returns
        -------
        None
            原地更新 ComboBox 选项或 QLineEdit 文本。

        Workflow
        --------
        遍历字段定义；枚举/布尔按 item data 定位，其他值调用统一格式化函数。
        """
        for section, key, _label, value_type in self._SETTING_FIELDS:
            editor = self.setting_editors[(section, key)]
            value = self.config[section][key]
            if value_type in {"bool", "filter"}:
                editor.setCurrentIndex(editor.findData(value))
            else:
                editor.setText(self._format_setting_value(value))

    def save_settings(self, *, show_message=True):
        """验证、原子保存并立即应用全部 UI 设置。

        Returns
        -------
        dict
            保存成功后的完整配置副本。

        Parameters
        ----------
        show_message : bool, keyword-only
            True 时用 QMessageBox 报告成功/失败；False 时失败异常直接向外抛出。

        Workflow
        --------
        收集并验证候选配置；原子写 config.json；成功后更新 Overview 页容量和窗口
        尺寸，向唯一 Detail 应用配置，重载第一页并规范化设置控件文本。
        """
        try:
            candidate = self.collect_settings_from_ui()
            save_config_json_atomic(self.config_path, candidate)
        except Exception as exc:
            if show_message:
                QMessageBox.critical(self, "设置保存失败", str(exc))
                return None
            raise

        self.config = candidate
        self.overview_config = self.config["overview"]
        self.rows_per_page = int(self.overview_config["rows_per_page"])
        self.cols_per_page = int(self.overview_config["cols_per_page"])
        self.items_per_page = self.rows_per_page * self.cols_per_page
        self.resize(
            int(self.overview_config["window_width"]),
            int(self.overview_config["window_height"]),
        )
        if self.detail_window is not None:
            self.detail_window.apply_config(self.config)
        self.current_page = 0
        self.load_page()
        self._refresh_setting_editors()
        if show_message:
            QMessageBox.information(self, "设置已保存", f"已保存到：\n{self.config_path}")
        return deepcopy(self.config)

    def _on_property_filter_changed(self, property_name):
        """响应一个动态 property 下拉框的变化。

        Parameters
        ----------
        property_name : str
            发生变化的真实 property 名称。

        Returns
        -------
        None
            将控件 data 交给 ``set_property_filter_state``。

        Workflow
        --------
        根据名称找到 QComboBox，读取当前 all/true/false data，并触发统一筛选流程。
        """
        combo = self.property_filter_combos[property_name]
        self.set_property_filter_state(property_name, combo.currentData())

    def set_property_filter_state(self, property_name, state):
        """设置单个属性的筛选状态并立即重新分页。

        Parameters
        ----------
        property_name : str
            Store 中实际存在的布尔 property 名称。
        state : {"all", "true", "false"}
            All 不限制该属性；True/False 只保留对应值。

        Returns
        -------
        None
            更新内存状态、控件和 Overview 页面。

        Workflow
        --------
        验证属性与枚举值；同步 filter_states；必要时阻断 signal 更新 ComboBox；
        最后执行多属性 AND 筛选。
        """
        if property_name not in self.properties:
            raise KeyError(f"不存在 property {property_name!r}。")
        if not isinstance(state, str) or state not in _VALID_FILTER_STATES:
            raise ValueError("筛选状态只能是 'all'、'true' 或 'false'。")

        self.filter_states[property_name] = state
        combo = self.property_filter_combos.get(property_name)
        if combo is not None and combo.currentData() != state:
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(state))
            combo.blockSignals(False)
        self.apply_property_filters()

    def _current_page_anchor_index(self):
        """取得筛选前用于保持查看位置的原始 epoch 锚点。

        Returns
        -------
        int or None
            优先返回当前页首个原始下标；没有当前页时返回当前可见位置；无可见项为 None。

        Workflow
        --------
        按 page_epoch_indices、visible_epoch_indices、空结果的优先顺序选择，不读 waveform。
        """
        if len(self.page_epoch_indices):
            return int(self.page_epoch_indices[0])
        if len(self.visible_epoch_indices):
            position = min(
                self.current_page * self.items_per_page,
                len(self.visible_epoch_indices) - 1,
            )
            return int(self.visible_epoch_indices[position])
        return None

    def apply_property_filters(self):
        """重新计算可见原始下标，并尽量保持当前查看位置。

        若锚点 epoch 仍然可见，新页面会包含它；若它被新条件隐藏，
        则选择原始下标数值上最近的可见 epoch。整个过程只生成 mask
        和下标数组；最后的 load_page() 仅读取新当前页 waveform。

        Returns
        -------
        None
            更新 visible_epoch_indices、current_page 和页面 Widget。

        Workflow
        --------
        保存原始下标锚点；构建 NumPy AND mask 和 flatnonzero；用 searchsorted 找仍
        可见或数值最近的下标；计算目标页并只加载该页 waveform。
        """
        anchor_index = self._current_page_anchor_index()
        mask = build_property_filter_mask(self.properties, self.filter_states)
        new_visible_indices = np.flatnonzero(mask).astype(np.int64, copy=False)

        if anchor_index is None or len(new_visible_indices) == 0:
            self.current_page = 0
        else:
            insertion_position = int(
                np.searchsorted(new_visible_indices, anchor_index, side="left")
            )
            candidate_positions = []
            if insertion_position < len(new_visible_indices):
                candidate_positions.append(insertion_position)
            if insertion_position > 0:
                candidate_positions.append(insertion_position - 1)
            nearest_position = min(
                candidate_positions,
                key=lambda position: (
                    abs(int(new_visible_indices[position]) - anchor_index),
                    position,
                ),
            )
            self.current_page = nearest_position // self.items_per_page

        self.visible_epoch_indices = new_visible_indices
        self.load_page()

    def _dirty_index_count(self):
        """统计全部 property 尚未保存的下标数量。

        Returns
        -------
        int
            各 ``set[int]`` 长度之和；同一 property 的重复修改只计一次。

        Workflow
        --------
        对 dirty_property_indices 的每个 set 取 len 并求和。
        """
        return sum(len(indices) for indices in self.dirty_property_indices.values())

    def _selected_add_property(self):
        """读取“添加属性”区域当前选择的属性名和值。

        Returns
        -------
        tuple[str or None, bool or None]
            当前属性名称和布尔值；任一未选择时返回对应的 ``None``。

        Workflow
        --------
        从两个下拉框读取 ``itemData``，不使用显示文本，确保布尔值保持真正的
        ``bool`` 类型，未选择项保持 ``None``。
        """
        property_name = self.add_property_name_combo.currentData()
        property_value = self.add_property_value_combo.currentData()
        return property_name, property_value

    def _update_add_property_button_state(self):
        """根据可写状态、选择数量和下拉框完整性更新按钮状态。

        Returns
        -------
        None
            直接更新 ``add_property_btn`` 的 enabled 状态。

        Workflow
        --------
        只有 Store 可写、至少选中一个 epoch、属性存在且值为 bool 时才启用按钮。
        """
        property_name, property_value = self._selected_add_property()
        enabled = (
            self.epochs_store.mode != "r"
            and bool(self.selected_epoch_indices)
            and property_name in self.properties
            and isinstance(property_value, (bool, np.bool_))
        )
        self.add_property_btn.setEnabled(enabled)

    def add_property_to_selected_epochs(self):
        """把选定的布尔属性值写入当前选中的全部 epoch（仅修改内存）。

        Returns
        -------
        int
            实际发生值变化并加入 dirty 集合的原始 epoch 下标数量。

        Workflow
        --------
        读取两个下拉框和已选原始下标；逐个记录首次旧值、更新内存 property 数组
        和 dirty 集合；重新应用筛选并刷新所有 Overview 状态。HDF5 只在用户点击
        “保存属性”后通过 ``save_dirty_properties`` 局部写入。
        """
        property_name, property_value = self._selected_add_property()
        if property_name not in self.properties or not isinstance(
                property_value,
                (bool, np.bool_),
        ):
            return 0
        if self.epochs_store.mode == "r" or not self.selected_epoch_indices:
            self._update_add_property_button_state()
            return 0

        property_value = bool(property_value)
        property_array = self.properties[property_name]
        dirty_indices = self.dirty_property_indices[property_name]
        original_values = self._dirty_original_values[property_name]
        changed_indices = []
        for original_index in sorted(self.selected_epoch_indices):
            original_index = int(original_index)
            if bool(property_array[original_index]) == property_value:
                continue
            if original_index not in dirty_indices:
                original_values[original_index] = bool(property_array[original_index])
            property_array[original_index] = property_value
            dirty_indices.add(original_index)
            changed_indices.append(original_index)

        if changed_indices:
            self.apply_property_filters()
            self.selected_epoch_indices.intersection_update(
                int(index) for index in self.visible_epoch_indices
            )
            self._refresh_current_page_selection()
        self.update_status_labels()
        return len(changed_indices)

    def delete_selected_epochs(self, *, confirm=True):
        """将已选 epoch 的 ``is_delete`` 属性设为 True，但不立即写入 HDF5。

        Parameters
        ----------
        confirm : bool, optional
            True 时显示确认对话框；测试或其他已确认流程可传 False。

        Returns
        -------
        int
            本次从 False 改为 True 的原始 epoch 数量。

        Workflow
        --------
        检查可写模式和选择；可选弹出确认框；逐个原始下标仅把 is_delete False 改为
        True，并记录首次原值和 dirty set；移除已更新选择并重新应用筛选。
        """
        if self.epochs_store.mode == "r":
            if confirm:
                QMessageBox.warning(self, "只读数据", "当前 HDF5 以只读模式打开，无法修改属性。")
            return 0
        selected_indices = sorted(self.selected_epoch_indices)
        if not selected_indices:
            if confirm:
                QMessageBox.information(self, "未选择", "请先选择要修改属性的 epoch。")
            return 0
        if confirm:
            answer = QMessageBox.question(
                self,
                "确认设置删除属性",
                f"确定将选中的 {len(selected_indices)} 个 epoch 的 is_delete 设置为 True 吗？\n"
                "waveform 不会被删除，点击“保存属性”后才会写入 HDF5。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return 0

        delete_values = self.properties["is_delete"]
        dirty_indices = self.dirty_property_indices["is_delete"]
        original_values = self._dirty_original_values["is_delete"]
        changed_indices = []
        for original_index in selected_indices:
            if not bool(delete_values[original_index]):
                if original_index not in dirty_indices:
                    original_values[original_index] = bool(delete_values[original_index])
                delete_values[original_index] = True
                dirty_indices.add(original_index)
                changed_indices.append(original_index)

        if changed_indices:
            self.selected_epoch_indices.difference_update(changed_indices)
            self.last_clicked_epoch_index = None
            self.apply_property_filters()
        else:
            self.update_status_labels()
        return len(changed_indices)

    def save_dirty_properties(self, *, show_message=True):
        """将 dirty properties 按原始下标局部写回 HDF5。

        主窗口先对 set 转换得到排序且去重的下标，Store 再将相邻
        下标合并为连续区间后写入。全部 property 写入并 flush 成功后
        才清空 dirty 状态，便于写入异常时重试。

        Returns
        -------
        int
            实际写入的 property-index 数量。

        Parameters
        ----------
        show_message : bool, keyword-only
            True 时显示“无需保存”或成功消息；False 用于关闭流程和自动化测试。

        Workflow
        --------
        将非空 dirty set 排序去重；逐 property 调用 Store 局部区间写入；全部成功
        后统一 flush；最后清空 dirty/原值记录并更新 UI。异常时保留 dirty 便于重试。
        """
        sorted_dirty = {
            property_name: sorted(set(indices))
            for property_name, indices in self.dirty_property_indices.items()
            if indices
        }
        if not sorted_dirty:
            if show_message:
                QMessageBox.information(self, "无需保存", "当前没有未保存的属性修改。")
            return 0

        written_count = 0
        for property_name, indices in sorted_dirty.items():
            written_count += self.epochs_store.write_property_indices(
                property_name,
                indices,
            )
        self.epochs_store.flush()

        for property_name in sorted_dirty:
            self.dirty_property_indices[property_name].clear()
            self._dirty_original_values[property_name].clear()
        self.update_status_labels()
        if show_message:
            QMessageBox.information(
                self,
                "保存成功",
                f"已保存 {written_count} 个属性下标。",
            )
        return written_count

    def discard_dirty_property_changes(self):
        """放弃尚未写入 HDF5 的内存属性修改。

        Returns
        -------
        int
            恢复的 property-index 数量。

        Workflow
        --------
        遍历首次修改前的原值字典，写回内存 property 数组并清空 dirty；有恢复时
        重做筛选，否则只更新状态标签；不会写 HDF5。
        """
        restored_count = 0
        for property_name, original_values in self._dirty_original_values.items():
            property_array = self.properties[property_name]
            for original_index, original_value in original_values.items():
                property_array[original_index] = original_value
                restored_count += 1
            original_values.clear()
            self.dirty_property_indices[property_name].clear()
        if restored_count:
            self.apply_property_filters()
        else:
            self.update_status_labels()
        return restored_count

    def _release_current_page(self):
        """释放当前 Overview 页的全部 Qt/Matplotlib/data 引用。

        Returns
        -------
        None
            清空网格、Widget 列表和 page_epoch_indices。

        Workflow
        --------
        从 QGridLayout 逐项取出；OverviewWidget 先释放 Figure/Canvas/waveform；所有
        Widget 脱离父对象并 deleteLater；最后清空轻量索引。
        """
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if isinstance(widget, OverviewWidget):
                widget.release_plot_resources()
            widget.setParent(None)
            widget.deleteLater()

        self.current_overview_widgets.clear()
        self.page_epoch_indices = np.empty(0, dtype=np.int64)

    def load_page(self):
        """逐条读取当前页 waveform 并创建 OverviewWidget。

        读取范围只来自 page_epoch_indices。该方法不会预读上一页、下一页或其他
        可见 epoch，也不会在主窗口中保存当前页之外的 waveform。

        Returns
        -------
        None
            更新当前页 Widget、page_epoch_indices、按钮和状态标签。

        Workflow
        --------
        释放旧页；计算当前可见下标切片；空页显示提示，否则逐原始下标调用一次
        get_waveform 并创建 Widget；补占位、设置伸缩比例并同步按钮。
        """
        self._release_current_page()
        self.calculate_total_pages()

        start = self.current_page * self.items_per_page
        end = min(start + self.items_per_page, len(self.visible_epoch_indices))
        self.page_epoch_indices = self.visible_epoch_indices[start:end].copy()

        if len(self.page_epoch_indices) == 0:
            empty_label = QLabel("暂无可显示的 epoch。")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet(
                "font-size: 18px; color: #9ca3af; margin: 100px;"
            )
            self.grid_layout.addWidget(
                empty_label,
                0,
                0,
                self.rows_per_page,
                self.cols_per_page,
            )
        else:
            for page_position, raw_index in enumerate(self.page_epoch_indices):
                original_epoch_index = int(raw_index)
                waveform = self.epochs_store.get_waveform(original_epoch_index)
                overview_widget = OverviewWidget(
                    original_epoch_index,
                    waveform,
                    self.epochs_store.epochs_ids[original_epoch_index],
                    self.config,
                    selected=(
                        original_epoch_index in self.selected_epoch_indices
                    ),
                    parent=self.grid_widget,
                )
                overview_widget.clicked.connect(self.on_overview_clicked)
                overview_widget.double_clicked.connect(
                    self.on_overview_double_clicked
                )
                row, column = divmod(page_position, self.cols_per_page)
                self.grid_layout.addWidget(overview_widget, row, column)
                self.current_overview_widgets.append(overview_widget)

            for page_position in range(
                    len(self.page_epoch_indices),
                    self.items_per_page,
            ):
                row, column = divmod(page_position, self.cols_per_page)
                placeholder = QLabel("")
                placeholder.setStyleSheet(
                    "background-color: transparent; border: none;"
                )
                self.grid_layout.addWidget(placeholder, row, column)

        for column in range(self.cols_per_page):
            self.grid_layout.setColumnStretch(column, 1)
        for row in range(self.rows_per_page):
            self.grid_layout.setRowStretch(row, 1)

        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < self.total_pages - 1)
        self.clear_page_btn.setEnabled(bool(len(self.page_epoch_indices)))
        self.last_clicked_epoch_index = None
        self.update_status_labels()

    def _visible_position(self, original_epoch_index):
        """查找一个原始 epoch 下标在当前可见数组中的位置。

        Parameters
        ----------
        original_epoch_index : Integral
            HDF5 中稳定的原始 epoch 下标。

        Returns
        -------
        int
            ``visible_epoch_indices`` 中对应的零基位置。

        Workflow
        --------
        用 NumPy 比较和 flatnonzero 查找；无匹配时抛出清楚错误。
        """
        matches = np.flatnonzero(
            self.visible_epoch_indices == int(original_epoch_index)
        )
        if len(matches) == 0:
            raise ValueError(
                f"原始 epoch 下标 {original_epoch_index} 不在当前可见结果中。"
            )
        return int(matches[0])

    def _update_overview_waveform_info(self, original_epoch_index):
        """在 Overview 顶栏显示一次单击对应的 epoch 基本信息。

        Parameters
        ----------
        original_epoch_index : Integral
            HDF5 中稳定的原始 epoch 下标。

        Returns
        -------
        None
            更新波形 ID、下标和 start_timestamp 三个标签；缺少时间戳时显示“未提供”。

        Workflow
        --------
        读取一个 epoch ID 和至多一个 start_timestamps 标量，使用五位有效数字格式化
        时间，然后把结果写入顶部三栏，不读取 waveform 数据。
        """
        original_epoch_index = int(original_epoch_index)
        epoch_id = self.epochs_store.epochs_ids[original_epoch_index]
        if self.epochs_store.start_timestamps is None:
            timestamp_text = "未提供"
        else:
            value = self.epochs_store.start_timestamps[original_epoch_index]
            value = value.item() if hasattr(value, "item") else value
            timestamp_text = f"{float(value):.5g}"
        self.waveform_id_label.setText(f"波形ID：{epoch_id}")
        self.waveform_index_label.setText(f"index：{original_epoch_index:,}")
        self.waveform_time_label.setText(f"时间：{timestamp_text}")

    def on_overview_clicked(self, original_epoch_index, shift_pressed=None):
        """按原始下标切换选择，或执行 Shift 连续选择。

        Parameters
        ----------
        original_epoch_index : int
            被点击 Widget 对应的原始 HDF5 epoch 下标。
        shift_pressed : bool or None, optional
            测试和非鼠标调用可显式指定；None 时读取当前 Qt 键盘修饰键。

        Returns
        -------
        None
            更新 selected_epoch_indices 和当前页选择样式。

        Workflow
        --------
        验证下标可见；确定 Shift 状态；Shift 时按可见数组位置加入闭区间，普通
        点击则切换单项；保存锚点并刷新当前页和状态标签。
        """
        original_epoch_index = int(original_epoch_index)
        self._visible_position(original_epoch_index)
        self._update_overview_waveform_info(original_epoch_index)

        if shift_pressed is None:
            shift_pressed = bool(
                QApplication.keyboardModifiers() & Qt.ShiftModifier
            )

        if shift_pressed and self.last_clicked_epoch_index is not None:
            anchor_position = self._visible_position(
                self.last_clicked_epoch_index
            )
            current_position = self._visible_position(original_epoch_index)
            start = min(anchor_position, current_position)
            end = max(anchor_position, current_position) + 1
            self.selected_epoch_indices.update(
                int(index)
                for index in self.visible_epoch_indices[start:end]
            )
        elif original_epoch_index in self.selected_epoch_indices:
            self.selected_epoch_indices.remove(original_epoch_index)
        else:
            self.selected_epoch_indices.add(original_epoch_index)

        self.last_clicked_epoch_index = original_epoch_index
        self._refresh_current_page_selection()
        self.update_status_labels()

    def _refresh_current_page_selection(self):
        """让当前页 Widget 显示主窗口中的跨页选择状态。

        Returns
        -------
        None
            逐个更新当前页 Widget 样式。

        Workflow
        --------
        遍历 current_overview_widgets，用其原始下标是否存在于全局选择 set 决定样式。
        """
        for overview_widget in self.current_overview_widgets:
            overview_widget.set_selected(
                overview_widget.original_epoch_index
                in self.selected_epoch_indices
            )

    def clear_all_selections(self):
        """清除全部页面选择，但不修改任何 property。

        Returns
        -------
        None
            清空选择 set、Shift 锚点，并刷新当前页与状态。

        Workflow
        --------
        清空 selected_epoch_indices，重置 last_clicked_epoch_index，再同步 UI。
        """
        self.selected_epoch_indices.clear()
        self.last_clicked_epoch_index = None
        self._refresh_current_page_selection()
        self.update_status_labels()

    def clear_current_page_selections(self):
        """只清除当前页原始下标的选择。

        Returns
        -------
        None
            其他页面的 selected_epoch_indices 保持不变。

        Workflow
        --------
        用 set.difference_update 移除 page_epoch_indices，重置 Shift 锚点并刷新 UI。
        """
        self.selected_epoch_indices.difference_update(
            int(index) for index in self.page_epoch_indices
        )
        self.last_clicked_epoch_index = None
        self._refresh_current_page_selection()
        self.update_status_labels()

    def _show_detail_warning(self, message):
        """使用非阻塞 QMessageBox 提示 Detail 数据缺失。

        非阻塞对话框不会启动嵌套 Qt 事件循环，因此自动化测试、
        脚本调用和普通鼠标双击都使用同一条安全路径。

        Parameters
        ----------
        message : str
            面向用户的中文缺失数据说明。

        Returns
        -------
        None
            QMessageBox 保存为 `_last_detail_warning_box` 供生命周期管理。

        Workflow
        --------
        创建非模态 Warning QMessageBox；设置关闭自动删除；offscreen 环境只创建
        不 show，真实平台显示，从而避免 Windows offscreen 底层崩溃。
        """
        warning_box = QMessageBox(self)
        warning_box.setIcon(QMessageBox.Warning)
        warning_box.setWindowTitle("无法打开 Continuous Detail")
        warning_box.setText(message)
        warning_box.setStandardButtons(QMessageBox.Ok)
        warning_box.setModal(False)
        warning_box.setAttribute(Qt.WA_DeleteOnClose, True)
        self._last_detail_warning_box = warning_box
        # Qt offscreen 插件在 Windows 上直接 show QMessageBox 可能引发底层
        # access violation。测试时保留完整 QMessageBox 供检查，真实界面才显示。
        if QApplication.platformName().lower() != "offscreen":
            warning_box.show()

    def _get_or_create_detail_window(self):
        """懒创建并返回 App 中唯一的 Detail 窗口。

        Returns
        -------
        DetailChartWindow
            现有实例，或首次调用时创建的新实例。

        Workflow
        --------
        detail_window 为 None 时用当前 continuous Store/config 创建并以主窗口为父；
        后续调用直接复用，不产生第二个窗口。
        """
        if self.detail_window is None:
            self.detail_window = DetailChartWindow(
                self.continuous_store,
                self.config,
                parent=self,
                epochs_store=self.epochs_store,
            )
        return self.detail_window

    def on_overview_double_clicked(self, original_epoch_index):
        """使用原始 epoch 的真实 start timestamp 打开唯一 Detail。

        Parameters
        ----------
        original_epoch_index : Integral
            双击 Widget 对应的 HDF5 原始 epoch 下标。

        Returns
        -------
        None
            通过 UI、signal 和 Detail 窗口表达结果。

        Workflow
        --------
        验证下标；只读取一个 start_timestamps 标量；更新双击信息并发送 signal；
        缺时间戳/continuous 时提示；否则取得唯一 Detail、设置中心时间并显示。
        """
        original_epoch_index = int(original_epoch_index)
        if original_epoch_index < 0 or original_epoch_index >= len(
                self.epochs_store
        ):
            raise IndexError(
                f"original_epoch_index {original_epoch_index} 超出有效范围。"
            )

        if self.epochs_store.start_timestamps is None:
            start_timestamp = None
            timestamp_text = "未提供"
        else:
            value = self.epochs_store.start_timestamps[original_epoch_index]
            start_timestamp = value.item() if hasattr(value, "item") else value
            timestamp_text = str(start_timestamp)

        self.last_double_click_info = (
            original_epoch_index,
            start_timestamp,
        )
        self._update_overview_waveform_info(original_epoch_index)
        self.double_click_label.setText(
            f"双击信息：原始 epoch 下标={original_epoch_index}，"
            f"start_timestamp={timestamp_text}"
        )
        self.overview_double_clicked.emit(
            original_epoch_index,
            start_timestamp,
        )

        if start_timestamp is None:
            self._show_detail_warning(
                "未传入 start_timestamps，无法定位连续数据。"
            )
            return
        if self.continuous_store is None:
            self._show_detail_warning(
                "未提供 continuous 数据，无法打开连续数据窗口。"
            )
            return

        detail_window = self._get_or_create_detail_window()
        detail_window.set_epoch_index(original_epoch_index)
        detail_window.show()
        detail_window.raise_()
        detail_window.activateWindow()

    def update_status_labels(self):
        """同步主窗口全部状态标签和操作按钮。

        Returns
        -------
        None
            更新总数、可见数、选择数、页码、dirty 数和按钮 enabled 状态。

        Workflow
        --------
        从轻量级索引/set 计算文本；仅 dirty>0 时允许保存；仅可写且有选择时允许删除。
        """
        self.info_label.setText(
            f"总 epoch：{len(self.epochs_store)} | "
            f"当前可见：{len(self.visible_epoch_indices)}"
        )
        self.selection_label.setText(
            f"已选择：{len(self.selected_epoch_indices)}"
        )
        self.page_label.setText(
            f"第 {self.current_page + 1} / {self.total_pages} 页"
        )
        dirty_count = self._dirty_index_count()
        self.dirty_label.setText(f"未保存：{dirty_count}")
        self.save_properties_btn.setEnabled(dirty_count > 0)
        self._update_add_property_button_state()

    def prev_page(self):
        """切换到上一 Overview 页。

        Returns
        -------
        None
            已在第一页时不操作。

        Workflow
        --------
        current_page>0 时减一并调用 load_page，只读取目标页 waveform。
        """
        if self.current_page > 0:
            self.current_page -= 1
            self.load_page()

    def next_page(self):
        """切换到下一 Overview 页。

        Returns
        -------
        None
            已在最后一页时不操作。

        Workflow
        --------
        未到末页时页码加一并调用 load_page，只读取目标页 waveform。
        """
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.load_page()

    def _close_owned_stores(self):
        """按 continuous、epoch 顺序幂等关闭本窗口拥有的 Store。

        Returns
        -------
        None
            非所有者、已经关闭或关闭完成均返回 None。

        Workflow
        --------
        检查 `_stores_closed` 和 `_owns_stores`；先设置关闭标记防止重入，再分别关闭
        可选 continuous Store 和必需 epoch Store。
        """
        if self._stores_closed or not self._owns_stores:
            return
        self._stores_closed = True
        if self.continuous_store is not None:
            self.continuous_store.close()
        if self.epochs_store is not None:
            self.epochs_store.close()

    def _delete_working_folder(self):
        """按启动选项删除工作目录及其文件。

        Returns
        -------
        None
            删除成功、目录已不存在或删除失败时均不向 Qt 传播异常。

        Workflow
        --------
        仅当 ``launch_epochs_verification_app`` 明确启用删除选项时执行；先确认
        自有 HDF5 句柄已关闭，再删除构造函数解析出的精确工作目录。删除失败时
        写入标准错误，避免关闭窗口因清理异常而崩溃。
        """
        if (
                not getattr(self, "_delete_working_folder_on_close", False)
                or getattr(self, "_working_folder_deleted", False)
        ):
            return
        try:
            self._close_owned_stores()
            folder = Path(self.working_folder).resolve()
            if folder.exists() and folder.is_dir():
                shutil.rmtree(folder)
            self._working_folder_deleted = True
        except Exception as exc:
            print(f"删除 working_folder 失败：{exc}", file=sys.stderr)

    def closeEvent(self, event):
        """处理未保存修改并统一释放全部 App 资源。

        Parameters
        ----------
        event : PyQt5.QtGui.QCloseEvent
            Qt 主窗口关闭事件，可被接受或忽略。

        Returns
        -------
        None
            通过 ``event.accept/ignore`` 返回关闭决定。

        Workflow
        --------
        有 dirty 时询问保存/放弃/取消；取消或保存失败则保持窗口和句柄；真正关闭时
        关闭警告框、释放唯一 Detail、当前 Overview 页和自有 Store，最后接受事件。
        """
        if self._dirty_index_count():
            answer = QMessageBox.question(
                self,
                "存在未保存修改",
                "properties 存在未保存修改，关闭前要如何处理？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Save:
                try:
                    self.save_dirty_properties(show_message=False)
                except Exception as exc:
                    QMessageBox.critical(
                        self,
                        "保存失败",
                        f"属性未能保存，窗口将继续保持打开：\n{exc}",
                    )
                    event.ignore()
                    return
            elif answer == QMessageBox.Discard:
                self.discard_dirty_property_changes()

        if not self._is_closing:
            self._is_closing = True
            if self._last_detail_warning_box is not None:
                self._last_detail_warning_box.close()
                self._last_detail_warning_box = None
            if self.settings_window is not None:
                self.settings_window.close()
            if self.detail_window is not None:
                self.detail_window.release_plot_resources()
                self.detail_window.close()
            self._release_current_page()
            self._close_owned_stores()
            self._delete_working_folder()
        event.accept()


def launch_epochs_verification_app(
        working_folder,
        epochs_dict=None,
        continuous_dict=None,
        config_dict=None,
        delete_working_folder=False,
        overwrite=None
):
    """创建并显示 App，同时安全复用已有 QApplication。

    外部脚本尚未创建 QApplication 时，本函数负责事件循环并返回退出码；在
    Jupyter、测试或已有 Qt 程序中调用时不会启动第二个事件循环，而是返回可继续
    操作的 ``DataViewer`` 实例。

    Returns
    -------
    int or DataViewer
        自己创建 QApplication 时返回事件循环退出码，否则返回主窗口实例。

    Parameters
    ----------
    working_folder : str or os.PathLike[str]
        保存三个固定文件的工作目录。
    epochs_dict : Mapping[str, object] or None, optional
        首次导入 epoch 数据；支持的 list/ndarray 形状、维度含义和缺省字段处理，
        详见 ``DataViewer.__init__`` 的 ``epochs_dict`` 参数。
    continuous_dict : Mapping[str, object] or None, optional
        首次导入动态 continuous 数据；通道/曲线嵌套结构、时间轴规则和缺省处理，
        详见 ``DataViewer.__init__`` 的 ``continuous_dict`` 参数。
    config_dict : Mapping[str, object] or None, optional
        用户局部配置；None 时自动加载 config.json 或默认配置。详见
        ``DataViewer.__init__`` 的 ``config_dict`` 参数。
    delete_working_folder : bool, optional
        是否在程序运行结束并关闭自有资源后删除整个 ``working_folder``。默认
        ``False``；设为 ``True`` 时适合临时数据目录，删除操作不可恢复。
    overwrite: bool or None
        写入时是否覆盖已有的文件，为None时跟随config文件的设置。

    Workflow
    --------
    检查是否已有 QApplication；必要时创建并设置 Fusion；构造、显示 DataViewer；
    自己创建 App 时进入事件循环并返回退出码，否则立即返回窗口实例。
    """
    qt_app = QApplication.instance()
    owns_application = qt_app is None
    if owns_application:
        qt_app = QApplication(sys.argv)
        qt_app.setStyle("Fusion")
    viewer = DataViewer(
        working_folder,
        epochs_dict=epochs_dict,
        continuous_dict=continuous_dict,
        config_dict=config_dict,
        overwrite=overwrite,
    )
    viewer._delete_working_folder_on_close = bool(delete_working_folder)
    if delete_working_folder:
        # 既有 QApplication 可能通过 app.quit() 结束而不发送窗口 closeEvent；
        # aboutToQuit 信号保证显式目录清理选项仍然生效。
        qt_app.aboutToQuit.connect(viewer._delete_working_folder)
    viewer.show()
    if owns_application:
        try:
            return qt_app.exec_()
        finally:
            # QApplication 可能在窗口 closeEvent 之外退出；此时仍需释放句柄后
            # 执行显式请求的目录清理。
            if delete_working_folder:
                viewer._release_current_page()
                viewer._close_owned_stores()
                viewer._delete_working_folder()
    return viewer


if __name__ == "__main__":
    # 使用临时工作目录，程序退出后自动清理示例生成的 HDF5 和 JSON 文件。
    # 固定随机种子可保证每次运行都得到相同数据，便于比较设置和绘图效果。
    demo_folder_handle = tempfile.TemporaryDirectory()
    demo_folder = Path(demo_folder_handle.name)
    demo_rng = np.random.default_rng(20260903)

    # 公共连续时间轴覆盖 120 秒，采样率为 500 Hz，共 60,001 个采样点。
    demo_sampling_rate = 500.0
    demo_timestamps = np.arange(60001, dtype=np.float64) / demo_sampling_rate

    # 创建 24 条长度不同、形态不同的 epoch 波形。每条 waveform 的一个点对应
    # 公共时间轴的一个采样点，因此 Detail 中标红的 epoch 区域长度具有实际意义。
    demo_epoch_count = 24
    demo_start_timestamps = np.linspace(5.0, 108.0, demo_epoch_count)
    demo_waveforms = []
    demo_has_artifact = np.zeros(demo_epoch_count, dtype=bool)
    for epoch_index in range(demo_epoch_count):
        waveform_length = 180 + (epoch_index * 137) % 521
        waveform_time = np.arange(waveform_length, dtype=np.float64) / demo_sampling_rate
        carrier_frequency = 4.0 + (epoch_index % 6) * 2.5
        amplitude = 0.8 + (epoch_index % 5) * 0.35
        envelope = np.exp(
            -((waveform_time - waveform_time.mean()) ** 2)
            / max(0.02, (waveform_time.max() * 0.32) ** 2)
        )
        waveform = (
            amplitude
            * envelope
            * np.sin(2.0 * np.pi * carrier_frequency * waveform_time)
            + 0.22 * np.sin(2.0 * np.pi * 1.2 * waveform_time + epoch_index * 0.3)
            + demo_rng.normal(0.0, 0.10, waveform_length)
        )

        # 每 6 条波形加入一个明显的瞬时伪迹，用于观察峰值与属性筛选效果。
        if epoch_index % 6 == 4:
            artifact_index = waveform_length // 2
            waveform[artifact_index:artifact_index + 4] += np.array(
                [3.8, -3.2, 2.5, -1.5]
            )
            demo_has_artifact[epoch_index] = True
        demo_waveforms.append(waveform)

    demo_high_amplitude = np.array(
        [np.max(np.abs(waveform)) >= 2.0 for waveform in demo_waveforms],
        dtype=bool,
    )
    demo_epochs = {
        "waveforms": demo_waveforms,
        "start_timestamps": demo_start_timestamps,
        "epochs_ids": [
            f"subject_A_trial_{epoch_index + 1:03d}"
            for epoch_index in range(demo_epoch_count)
        ],
        "properties": {
            "is_delete": np.isin(np.arange(demo_epoch_count), [3, 16]),
            "人工确认": np.arange(demo_epoch_count) % 3 != 1,
            "高振幅": demo_high_amplitude,
            "包含伪迹": demo_has_artifact,
        },
    }

    # 多个高频通道使用公共时间轴；同一通道的多条曲线会画在同一个子图中。
    theta_wave = np.sin(2.0 * np.pi * 7.0 * demo_timestamps)
    slow_wave = np.sin(2.0 * np.pi * 0.35 * demo_timestamps)
    stimulation = np.where(
        (demo_timestamps % 20.0 >= 8.0) & (demo_timestamps % 20.0 <= 12.0),
        1.0,
        0.0,
    )
    ecg_phase = demo_timestamps % 0.82

    # 运动数据和温度数据采用各自的低频时间轴，用来演示独立 timestamps。
    movement_timestamps = np.arange(6001, dtype=np.float64) / 50.0
    movement_speed = np.maximum(
        0.0,
        5.0
        + 3.0 * np.sin(2.0 * np.pi * 0.08 * movement_timestamps)
        + demo_rng.normal(0.0, 0.35, movement_timestamps.size),
    )
    temperature_timestamps = np.arange(1201, dtype=np.float64) / 10.0

    demo_continuous = {
        "__common_timestamps__": demo_timestamps,
        "前额叶 LFP": {
            "raw": {
                "values": (
                    1.1 * theta_wave
                    + 0.35 * slow_wave
                    + demo_rng.normal(0.0, 0.18, demo_timestamps.size)
                ),
            },
            "theta_filtered": {"values": 1.1 * theta_wave},
        },
        "海马 LFP": {
            "raw": {
                "values": (
                    0.75 * np.sin(2.0 * np.pi * 9.0 * demo_timestamps + 0.7)
                    + 0.28 * slow_wave
                    + demo_rng.normal(0.0, 0.16, demo_timestamps.size)
                ),
            },
            "slow_component": {"values": 0.28 * slow_wave},
        },
        "视觉皮层 LFP": {
            "raw": {
                "values": (
                    0.6 * np.sin(2.0 * np.pi * 12.0 * demo_timestamps)
                    + 0.9 * stimulation
                    + demo_rng.normal(0.0, 0.14, demo_timestamps.size)
                ),
            },
            "stimulus_state": {"values": stimulation},
        },
        "心电 ECG": {
            "lead_I": {
                "values": (
                    0.08 * np.sin(2.0 * np.pi * 1.22 * demo_timestamps)
                    + 1.8 * np.exp(-((ecg_phase - 0.10) / 0.018) ** 2)
                    - 0.45 * np.exp(-((ecg_phase - 0.14) / 0.025) ** 2)
                ),
            },
        },
        "呼吸 Respiration": {
            "airflow": {
                "values": (
                    1.4 * np.sin(2.0 * np.pi * 0.24 * demo_timestamps)
                    + 0.12 * np.sin(2.0 * np.pi * 0.48 * demo_timestamps)
                ),
            },
        },
        "肌电 EMG": {
            "raw": {
                "values": (
                    0.18 * np.sin(2.0 * np.pi * 45.0 * demo_timestamps)
                    + demo_rng.normal(0.0, 0.12, demo_timestamps.size)
                ),
            },
            "amplitude_envelope": {
                "values": 0.16 + 0.10 * (slow_wave + 1.0),
            },
        },
        "运动 Movement": {
            "speed": {
                "timestamps": movement_timestamps,
                "values": movement_speed,
            },
            "acceleration": {
                "timestamps": movement_timestamps,
                "values": np.gradient(movement_speed, movement_timestamps),
            },
        },
        "体温 Temperature": {
            "temperature": {
                "timestamps": temperature_timestamps,
                "values": (
                    37.0
                    + 0.18 * np.sin(2.0 * np.pi * temperature_timestamps / 90.0)
                    + demo_rng.normal(0.0, 0.015, temperature_timestamps.size)
                ),
            },
        },
    }

    # Overview 每页显示 12 条波形；Detail 每页约显示 3 个通道，便于测试两种分页。
    demo_config = {
        "overview": {
            "rows_per_page": 3,
            "cols_per_page": 4,
            "window_width": 1500,
            "window_height": 950,
            "subplot_min_height": 190,
            "line_width": 0.9,
        },
        "detail": {
            "window_width": 1500,
            "window_height": 850,
            "figure_width": 1400,
            "subplot_height": 220,
            "time_before": 2.0,
            "time_after": 4.0,
            "slider_points_per_second": 5000,
            "max_points_per_pixel": 1.5,
        },
    }
    try:
        raise SystemExit(
            launch_epochs_verification_app(
                demo_folder,
                epochs_dict=demo_epochs,
                continuous_dict=demo_continuous,
                config_dict=demo_config,
                delete_working_folder=True,
            )
        )
    finally:
        demo_folder_handle.cleanup()
