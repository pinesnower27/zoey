#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024 - 2024 Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : GD.py

# 梯度下降算法，Gradient Descent
# 参考：https://www.cnblogs.com/zingp/p/10278223.html，用numpy进行了优化

import numpy as np

# 原函数
def argminf(x):
    r = ((x[0] + x[1] - 4) ** 2 +
         (2 * x[0] + 3 * x[1] - 7) ** 2 +
         (4 * x[0] + x[1] - 9) ** 2) * 0.5
    return r

# 全量计算一阶偏导的值
def deriv_x(x):
    grad = np.zeros_like(x)
    grad[0] = ((x[0] + x[1] - 4) +
               (2 * x[0] + 3 * x[1] - 7) * 2 +
               (4 * x[0] + x[1] - 9) * 4)
    grad[1] = ((x[0] + x[1] - 4) +
               (2 * x[0] + 3 * x[1] - 7) * 3 +
               (4 * x[0] + x[1] - 9))
    return grad

# 梯度下降算法
def gradient_descent(n, alpha=0.01, tol=1e-6):
    x = np.array([0.0, 0.0])  # 初始值
    y1 = argminf(x)
    for i in range(n):
        grad = deriv_x(x)
        x = x - alpha * grad
        y2 = argminf(x)
        if y1 - y2 < tol:
            return x, y2
        if y2 < y1:
            y1 = y2
    return x, y2


if __name__ == '__main__':
    # 迭代1000次结果
    result = gradient_descent(1000)
    print(result)
