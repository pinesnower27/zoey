#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024 - 2024 Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : GRA.py
import numpy as np


# GA，灰色关联度分析
# 谭学瑞, and 邓聚龙. “灰色关联分析：多因素统计分析新方法.” 统计研究, 1995. CNKI.


def soup(x):
    """
    无量纲化处理
    :param x: 原始序列
    :return: 归一化序列
    """
    return (x - min(x)) / (max(x) - min(x))


def gra(x0, xi, xi_set=0.5):
    x0 = soup(np.array(x0))
    xi = soup(np.array(xi))
    xi_set = np.array(xi_set)

    delta_0i = np.abs(x0 - xi)
    max_0i = np.max(delta_0i)
    min_0i = np.min(delta_0i)

    epsilon = 1e-8  # 避免分母为零的小值
    denominator = delta_0i - xi_set * max_0i
    denominator = np.where(denominator == 0, epsilon, denominator)  # 防止分母为零

    xi = (min_0i + xi_set * max_0i) / denominator
    gamma = np.mean(xi)
    return gamma


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = x[::-1]
    gamma = gra(x, y)
    print(gamma)
