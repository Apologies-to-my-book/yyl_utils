"""
yyl_utils 的统一导入入口。

导入 yyl_utils 时，只创建公开名称与模块之间的映射，
不会立即导入各个功能模块。

访问公开函数、类或子模块时，才动态导入对应对象。
"""

from importlib import import_module


__version__ = "1.0.0"


# 每个模块需要对外公开的函数和类名称。
_EXPORTS = {
    ".debug_utils": """
        ipdb_debug
        decorator_func_error
        decorator_func_profile
        ipdb_trace
        interactive_method_explorer
        detailed_method_analysis
        get_class_mro
        detailed_class_info
        redirect_print_to_log
        analyze_obj_memory
        get_full_size
        print_dir
    """.split(),

    ".plot_utils": """
        set_backend
        set_chinese_font
        plot_CI_map
        get_color_with_opencv
        fast_plot_psd
        LargeDataSlidePlot
        plot_units_waveforms
    """.split(),

    ".other_utils": """
        check_delete_exists_path
        matlab_struct_to_dict
        isolate
        make_sure_folder_exist
        add_suffix_to_filename
        delete_path
        SettingsWithSave
        IntervalOps
    """.split(),

    ".spike_sorting_utils": """
        SpikeSortingPipeline
        launch_phy
    """.split(),

    ".lfp_utils": """
        RippleDetector
        calc_heatmap_PAC
        calc_psd
        calc_band_psd_by_simpson
        SOSFilter
        fit_log_pink_noise
    """.split(),

    ".waveclus_python": """
        WaveClusBatchSorter
    """.split(),

    ".epochs_verification_app": """
        launch_epochs_verification_app
    """.split(),
}


# 将函数名或类名转换为“名称 -> 所在模块”的映射。
_NAME_TO_MODULE = {
    name: module_name
    for module_name, names in _EXPORTS.items()
    for name in names
}


# 需要允许通过 yyl_utils.子模块名 访问的子模块。
# 这里的子模块仍然是延迟导入，不会在 import yyl_utils 时立即加载。
_LAZY_MODULES = {
    "debug_utils": ".debug_utils",
    "epochs_verification_app": ".epochs_verification_app",
    "file_reader_utils": ".file_reader_utils",
    "lfp_utils": ".lfp_utils",
    "other_utils": ".other_utils",
    "plot_utils": ".plot_utils",
    "spike_sorting_utils": ".spike_sorting_utils",
    "waveclus_python": ".waveclus_python",
}


# 同时公开函数、类和子模块名称。
__all__ = list(_NAME_TO_MODULE) + list(_LAZY_MODULES)


def __getattr__(name):
    """
    第一次访问公开函数、类或子模块时，动态导入对应对象。

    Parameters
    ----------
    name : str
        要访问的函数名、类名或子模块名。

    Returns
    -------
    object
        对应的函数、类或模块对象。

    Raises
    ------
    AttributeError
        当 yyl_utils 中不存在指定名称时抛出。

    Workflow
    --------
    1. 先判断 name 是否为公开子模块。
    2. 如果是，则动态导入并返回该子模块。
    3. 如果不是子模块，再判断 name 是否为公开函数或类。
    4. 动态导入函数或类所在的模块并取得目标对象。
    5. 将导入结果缓存到当前包，后续访问时不再重复导入。
    """

    # 处理：yyl_utils.epochs_verification_app
    module_name = _LAZY_MODULES.get(name)
    if module_name is not None:
        module = import_module(
            module_name,
            package=__name__,
        )

        # 缓存子模块对象，后续访问时不再重复导入。
        globals()[name] = module
        return module

    # 处理：yyl_utils.SOSFilter
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(
            f"模块 {__name__!r} 没有属性 {name!r}"
        )

    module = import_module(
        module_name,
        package=__name__,
    )
    value = getattr(module, name)

    # 缓存函数或类对象，后续访问时不再重复执行动态导入。
    globals()[name] = value
    return value


def __dir__():
    """
    返回 yyl_utils 对外公开的名称。

    Returns
    -------
    list[str]
        包含函数、类和子模块名称的排序列表。
    """
    return sorted(
        set(globals())
        | set(__all__)
        | set(_LAZY_MODULES)
    )
