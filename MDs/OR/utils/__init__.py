import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def bootstrap(data, n=5000, is_trans=False, is_header=False):
    """
    实现功能：传入参数data，返回重复n次抽样后的汇总结果
    :param data: 传入的原始数组
    :param n: 重新抽样次数
    :param is_trans: 是否转置
    :param is_header: 是否有头
    :return: 汇总结果
    """
    # 预处理
    if is_header:
        header = data[0, :]
        data = data[1:, :]
    if is_trans:
        data = np.array(data).T
    data = np.array(data)

    # 重抽样实现
    bootstrap_array = np.empty(n)  # 生成空的np数据
    value_num, feature_num = data.shape
    for i in range(n):
        # 从数据集中随机抽取样本
        sample_indices = np.random.choice(value_num, size=value_num, replace=True)
        sample = data[sample_indices, :]  # 使用这些索引生成新的样本
        # 对抽样样本计算感兴趣的统计量，这里以均值为例
        sample_stats = np.mean(sample, axis=0)  # 计算每列的均值
        bootstrap_array = np.vstack([bootstrap_array, sample_stats]) if i > 0 else np.array([sample_stats])

    # 汇总结果
    if is_header:
        bootstrap_results = np.vstack([header, bootstrap_array])  # 如果有头，添加头信息
    else:
        bootstrap_results = bootstrap_array

    return bootstrap_results


def feature_bootstrap(data, header=None, n=5000, is_trans=False):
    """
    针对某一列的重抽样（TODO: 需要修改）
    :param data: 原始数据
    :param header: 头的名字
    :param n: 重复的次数
    :param is_trans: 是否转置
    :return: 汇总结果
    """
    data = np.array(data)
    if is_trans:
        data = data.T
    # 确定目标列
    if header is not None:  # 如果传入了 header
        if is_trans:
            target_feature_idx = np.where(data[0, :] == header)[0][0]  # 转置后标题在第一行
        else:
            target_feature_idx = np.where(data[:, 0] == header)[0][0]  # 标题在第一列
    else:  # 如果没有 header，默认使用第一列
        target_feature_idx = 0

    # 提取目标列（行）进行重抽样
    if is_trans:
        target_feature = data[1:, target_feature_idx]  # 转置后标题在第一行，数据从第二行开始
    else:
        target_feature = data[target_feature_idx, 1:]  # 数据在第一列，跳过标题行/列

    target_feature = target_feature.astype(float)  # 确保数据为数值型
    result = np.empty((n, len(target_feature)))  # 用于存储抽样结果

    for i in range(n):
        sampled_indices = np.random.choice(len(target_feature), size=len(target_feature), replace=True)
        result[i] = target_feature[sampled_indices]  # 重抽样的结果

    return result


if __name__ == '__main__':
    np.random.seed(42)
    dta = np.random.random((10, 20))
    rb = bootstrap(dta)
    print(rb.shape)


