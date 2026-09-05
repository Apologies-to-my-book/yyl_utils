def get_nex5_continuous(
    nex5_path,
    channel_indices=tuple(range(10)),
    time_arange=None,
    n_jobs=4,
):
    """
    读取 NEX5 文件中指定 continuous 通道的信号。

    Parameters
    ----------
    nex5_path : str | Path
        NEX5 文件路径。
    channel_indices : list[int] | tuple[int, ...], default: tuple(range(10))
        需要读取的 continuous 通道索引，同时决定输出矩阵中的通道顺序。

        例如传入 ``[0, 2, 5]`` 时，返回矩阵的三列依次对应 continuous
        通道列表中的第 0、2、5 个通道。

    time_arange : tuple[int, int] | list[int] | None, default: None
        需要返回的采样点范围，格式为 ``[start, stop]``，采用左闭右开区间：

        - ``start``：起始采样点，包含该采样点。
        - ``stop``：结束采样点，不包含该采样点。
        - ``None``：返回完整时间范围。

        例如 ``[1000, 2000]`` 返回第 1000～1999 个采样点。

        注意：当前 nex5file 的公开读取方法仍然会先读取完整通道，
        然后本函数再截取指定范围。因此该参数可以缩小最终输出矩阵，
        但不能减少 nex5file 读取完整通道时的内存和时间。

    n_jobs : int, default: 4
        Joblib 并行读取线程数：

        - ``1``：依次读取各个通道。
        - 大于 ``1``：同时读取多个通道。
        - ``-1``：使用全部可用逻辑 CPU。

        并行数越大，同时存在于内存中的完整通道数组越多，因此大型
        NEX5 文件不建议直接使用 ``-1``。

    Returns
    -------
    dict
        返回包含以下字段的字典：

        - ``"traces"``：``numpy.ndarray``，形状为
          ``(n_samples, n_selected_channels)``，数据类型为 ``float32``，
          内存布局为 F-order，单位为 μV。
        - ``"fs"``：``float``，采样频率，单位为 Hz。
        - ``"chan_ids"``：``list[str]``，输出矩阵每一列对应的通道名称。

    Workflow
    --------
    1. 读取 NEX5 文件头，获取 continuous 通道名称和采样频率。
    2. 根据 ``channel_indices`` 取得需要读取的通道名称。
    3. 创建 F-order 的最终输出矩阵，使每一个通道列在内存中连续。
    4. 使用 Joblib 多线程读取各个完整通道。
    5. 截取指定采样点范围，将 mV 转换为 μV，并直接写入输出矩阵。
    6. 返回波形矩阵、采样率和通道名称。
    """
    import numpy as np
    from joblib import Parallel, delayed
    from nex5file.reader import Reader

    # 只读取文件头，取得 continuous 通道的名称和基本信息。
    header_data = Reader().ReadNex5HeadersOnly(nex5_path)
    continuous_names = header_data.ContinuousNames()
    channel_names = [
        continuous_names[index]
        for index in channel_indices
    ]

    first_channel = header_data[channel_names[0]]
    fs = first_channel.header.SamplingRate

    if time_arange is None:
        start = 0
        stop = first_channel.header.NPointsWave
    else:
        start, stop = time_arange

    # F-order 使 traces 的每一列连续存储。
    # 当前函数按通道逐列写入，因此这种布局更适合当前的数据填充方式。
    traces = np.empty(
        (stop - start, len(channel_names)),
        dtype=np.float32,
        order="F",
    )

    def read_channel(channel_info):
        """
        读取一个完整 continuous 通道并写入最终矩阵。

        Parameters
        ----------
        channel_info : tuple[int, str]
            包含两个值：

            - ``output_column``：该通道在输出矩阵中的列索引。
            - ``channel_name``：NEX5 文件中的通道名称。

        Returns
        -------
        None
            数据直接写入共享的 ``traces``，不单独返回数组。

        Workflow
        --------
        1. 使用 nex5file 读取指定通道。
        2. 取得单位为 mV 的完整通道数组。
        3. 使用基本切片取得目标采样点范围。
        4. 将数据乘以 1000 转换为 μV。
        5. 直接写入 ``traces`` 对应的连续列。
        """
        output_column, channel_name = channel_info

        channel_data = Reader().ReadNex5FileVariables(
            nex5_path,
            [channel_name],
        )
        values_mv = channel_data.variables[0].continuous_values

        # values_mv[start:stop] 是 NumPy 视图，不会复制完整通道。
        # out 指定最终输出列，因此乘法结果也不会生成额外的大数组。
        np.multiply(
            values_mv[start:stop],
            1000.0,
            out=traces[:, output_column],
            casting="unsafe",
        )

    Parallel(
        n_jobs=n_jobs,
        backend="threading",
    )(
        delayed(read_channel)(channel_info)
        for channel_info in enumerate(channel_names)
    )

    return {
        "traces": traces,
        "fs": fs,
        "chan_ids": channel_names,
    }
