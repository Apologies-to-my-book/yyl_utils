# my_utils/__init__.py

# 版本信息
__version__ = "1.0.0"

# 从其他文件导入函数
from .debug_utils import ipdb_debug, decorator_func_error, decorator_func_profile, ipdb_trace, \
    interactive_method_explorer, detailed_method_analysis, get_class_mro,\
    detailed_class_info,redirect_print_to_log,analyze_obj_memory,get_full_size
from .plot_utils import set_backend,set_chinese_font,plot_CI_map,get_color_with_opencv,fast_plot_psd, \
    LargeDataSlidePlot
from .other_utils import check_delete_exists_path,matlab_struct_to_dict,isolate,make_sure_folder_exist,\
    add_suffix_to_filename,delete_path,SettingsWithSave,IntervalOps
from .spikeinterface_utils import print_sorter_params,SpikeSortingPipeline,launch_phy
from .lfp_utils import RippleDetector,calc_heatmap_PAC,calc_psd,calc_band_psd_by_simpson,SOSFilter,\
    fit_log_pink_noise

# 定义 * 导入的内容
__all__ = []
# .debug_utils导入
__all__.extend([
    'ipdb_debug',
    'decorator_func_error',
    'decorator_func_profile',
    'ipdb_trace',
    'interactive_method_explorer',
    'detailed_method_analysis',
    'get_class_mro',
    'detailed_class_info',
    'redirect_print_to_log',
    'analyze_obj_memory',
    'get_full_size',
])
# .plot_utils导入
__all__.extend([
    'set_backend',
    'set_chinese_font'
    ,'plot_CI_map',
    'fast_plot_psd',
    'LargeDataSlidePlot',
])
# .other_utils导入
__all__.extend([
    'check_delete_exists_path'
    ,'matlab_struct_to_dict',
    'isolate',
    'make_sure_folder_exist',
    'get_color_with_opencv',
    'add_suffix_to_filename',
    'delete_path',
    'SettingsWithSave',
    'IntervalOps',
])
# .spikeinterface_utils导入
__all__.extend([
    'print_sorter_params',
    'SpikeSortingPipeline',
    'launch_phy',
])
# .lfp_utils导入
__all__.extend([
    'RippleDetector',
    'calc_heatmap_PAC',
    'calc_psd',
    'calc_band_psd_by_simpson',
    'SOSFilter',
    'fit_log_pink_noise',
])