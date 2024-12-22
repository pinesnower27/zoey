#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024 - 2024 Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : OLS.py

# OLS, 最小二乘估计
"""TODO:
1. [ ] 请重构该class的所有函数
2. [ ] 使用最小二乘估计的方法计算回归方程，残差项，t值等放在self._summary字典里。
"""

import numpy as np
import pandas as pd


class OLS:
    def __init__(self, y, x):
        self.y = np.array(y) # 因变量
        self.x = np.array(x) # 自变量
        self.reg_parameters = np.array([]) # 回归系数
        self._summary: dict = {} # 存储模型的统计摘要

    def fit(self):
        """
        这里是实现拟合，然后获得回归系数的函数
        :return:
        """
        # 如果自变量没有常数项，加入一列常数项（全为1的列）
        if self.x.ndim == 1:
            X = np.vstack([np.ones_like(self.x), self.x]).T  # 添加常数项
        else:
            X = np.column_stack([np.ones(len(self.x)), self.x])  # 如果有多列x, 添加常数项

        # 计算回归系数
        X_transpose = X.T
        X_inv = np.linalg.inv(X_transpose @ X)  # (X^T X)^-1
        betas = X_inv @ X_transpose @ self.y  # 计算beta系数

        # 保存回归系数
        self.reg_parameters = betas

        # 预测值
        y_hat = X @ betas

        # 计算残差
        residuals = self.y - y_hat

        # 计算误差的标准差（标准误差）
        n, k = X.shape  # n: 样本数, k: 自变量个数
        sigma_squared = (residuals.T @ residuals) / (n - k)  # 方差
        standard_errors = np.sqrt(np.diagonal(sigma_squared * np.linalg.inv(X_transpose @ X)))  # 标准误差

        # 计算 t 值
        t_values = betas / standard_errors

        # 计算R-squared
        ss_total = np.sum((self.y - np.mean(self.y)) ** 2)  # 总平方和
        ss_residual = np.sum(residuals ** 2)  # 残差平方和
        r_squared = 1 - ss_residual / ss_total  # R-squared

        # 统计摘要
        self._summary = {
            'coefficients': betas,
            'standard_errors': standard_errors,
            't_values': t_values,
            'r_squared': r_squared,
            'residuals': residuals
        }


    def betas(self):
        return self.reg_parameters

    def predict(self, x_predict):
        """
        基于已有的进行预测，传入是x的数据，然后返回预测值y
        :param x_predict: 未来自变量的value
        :return: 预测出的value
        """
        if x_predict.ndim == 1:
            X_predict = np.vstack([np.ones_like(x_predict), x_predict]).T
        else:
            X_predict = np.column_stack([np.ones(len(x_predict)), x_predict])

        return X_predict @ self.betas()

    def summary(self):
        return self._summary


if __name__ == '__main__':
    # 读取数据
    data_path = "../../../src/dta/cat_dog.csv"
    df = pd.read_csv(data_path)

    # 下面应该实现Stata中`reg cat dog`相同的功能
    ols = OLS(y=df["cat"], x=df["dog"])

    # 拟合模型
    ols.fit()

    # 进行预测
    x_pred = df["dog"]
    y_pred = ols.predict(x_predict=x_pred)

    # 输出回归结果摘要
    print(ols.summary())
