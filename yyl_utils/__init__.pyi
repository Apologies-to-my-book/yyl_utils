"""
yyl_utils 的静态类型声明文件。

本文件供 VS Code、Pylance 和其他类型检查工具使用。
它声明根包公开的函数、类和子模块，使以下写法能够自动补全：

    import yyl_utils
    yyl_utils.epochs_verification_app.launch_epochs_verification_app

运行时的实际延迟导入逻辑由同目录的 __init__.py 负责。
"""

from . import debug_utils as debug_utils
from . import epochs_verification_app as epochs_verification_app
from . import file_reader_utils as file_reader_utils
from . import lfp_utils as lfp_utils
from . import other_utils as other_utils
from . import plot_utils as plot_utils
from . import spike_sorting_utils as spike_sorting_utils
from . import waveclus_python as waveclus_python

from .debug_utils import (
    analyze_obj_memory as analyze_obj_memory,
    decorator_func_error as decorator_func_error,
    decorator_func_profile as decorator_func_profile,
    detailed_class_info as detailed_class_info,
    detailed_method_analysis as detailed_method_analysis,
    get_class_mro as get_class_mro,
    get_full_size as get_full_size,
    interactive_method_explorer as interactive_method_explorer,
    ipdb_debug as ipdb_debug,
    ipdb_trace as ipdb_trace,
    print_dir as print_dir,
    redirect_print_to_log as redirect_print_to_log,
)

from .epochs_verification_app import (
    launch_epochs_verification_app as launch_epochs_verification_app,
)

from .lfp_utils import (
    RippleDetector as RippleDetector,
    SOSFilter as SOSFilter,
    calc_band_psd_by_simpson as calc_band_psd_by_simpson,
    calc_heatmap_PAC as calc_heatmap_PAC,
    calc_psd as calc_psd,
    fit_log_pink_noise as fit_log_pink_noise,
)

from .other_utils import (
    IntervalOps as IntervalOps,
    SettingsWithSave as SettingsWithSave,
    add_suffix_to_filename as add_suffix_to_filename,
    check_delete_exists_path as check_delete_exists_path,
    delete_path as delete_path,
    isolate as isolate,
    make_sure_folder_exist as make_sure_folder_exist,
    matlab_struct_to_dict as matlab_struct_to_dict,
)

from .plot_utils import (
    LargeDataSlidePlot as LargeDataSlidePlot,
    fast_plot_psd as fast_plot_psd,
    get_color_with_opencv as get_color_with_opencv,
    plot_CI_map as plot_CI_map,
    plot_units_waveforms as plot_units_waveforms,
    set_backend as set_backend,
    set_chinese_font as set_chinese_font,
)

from .spike_sorting_utils import (
    SpikeSortingPipeline as SpikeSortingPipeline,
    launch_phy as launch_phy,
)

from .waveclus_python import (
    WaveClusBatchSorter as WaveClusBatchSorter,
)

__version__: str
__all__: list[str]
