def plot_before_and_after_data(before_data, after_data,title_name,x_axis=None,mode='recording'):
    '''
    给处理前后的数据画图，最终会画在一个图内,会生成html
    :param x_axis: 画图的x轴
    :param before_data:
    :param after_data:
    :param title_name: 标题名字
    :return:
    '''
    import panel as pn
    import holoviews as hv
    import pandas as pd
    import numpy as np
    import os
    import gc

    pn.extension(sizing_mode="stretch_width")
    def create_signal_app(signal1, signal2, t):
        '''
        画图，生成一个html对象，图中会有2个线，分别是处理前和处理后的
        :param signal1: 处理前信号
        :param signal2: 处理后信号
        :param t: 横轴
        :return:
        '''
        # 创建数据帧
        df1 = pd.DataFrame({'time': t, 'signal': signal1.flatten()})
        df2 = pd.DataFrame({'time': t, 'signal': signal2.flatten()})
        # 创建 Holoviews 图表
        curve1 = hv.Curve(df1, 'time', 'signal', label='before').opts(color="blue", line_width=2)
        curve2 = hv.Curve(df2, 'time', 'signal', label='after').opts(color="red", line_width=2)
        curve = (curve1 * curve2).opts(title=f"{title_name}前后的信号对比",legend_position='top_left')
        # 将 Holoviews 图表转换为 Panel pane
        plot_panel = pn.pane.HoloViews(
            curve,
            width=1200,  # 强制设定宽度
            height=800  # 强制设定高度
        )
        # 布局
        app = pn.Row(
            plot_panel,
            sizing_mode='stretch_both',  # 布局随页面大小拉伸
            width_policy='max',  # 最大化宽度
            height_policy='max'  # 最大化高度
        )
        return app
    if mode=='recording':
        channel_ids = before_data.get_channel_ids()[0]
        before_data=before_data.get_traces(channel_ids=[channel_ids])
        after_data=after_data.get_traces(channel_ids=[channel_ids])
    # 定义x轴
    if x_axis is None:
        x_axis=np.arange(len(before_data))/30000
    nperseg=2000000
    for i in np.arange(1,np.ceil(len(x_axis)/nperseg)):
        i=int(i)
        plot_before_data=before_data[(i-1)*nperseg:i*nperseg].copy()
        plot_after_data=after_data[(i-1)*nperseg:i*nperseg].copy()
        plot_axis=x_axis[(i-1)*nperseg:i*nperseg].copy()
        app=create_signal_app(plot_before_data, plot_after_data,plot_axis)
        # 保存为HTML文件
        html_file =os.path.join(r"C:\Users\32707\Desktop",f"测试-{title_name}_{i}.html")
        app.save(html_file)
    del app, plot_before_data, plot_after_data, plot_axis,after_data,before_data
    gc.collect()  # 强制垃圾回收