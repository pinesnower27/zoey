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
        self.y = np.array(y)
        self.x = np.array(x)
        self.reg_parameters = np.array([])
        self._summary: dict = {}

    def fit(self):
        """
        这里是实现拟合，然后获得回归系数的函数
        :return:
        """
        pass

    def betas(self):
        return self.reg_parameters

    def predict(self, x_predict):
        """
        基于已有的进行预测，传入是x的数据，然后返回预测值y
        :param x_predict: 未来自变量的value
        :return: 预测出的value
        """
        y_predict = x_predict
        betas = self.betas()
        return y_predict

    def summary(self):
        return self._summary


if __name__ == '__main__':
    data_path = "../../src/dta/cat_dog.csv"
    df = pd.read_csv(data_path)
    # 下面应该实现Stata中`reg cat dog`相同的功能
    ols = OLS(y=df["cat"], x=df["dog"])
    ols.fit()
    x_pred = df["dog"]
    y_pred = ols.predict(x_predict=x_pred)
    print(ols.summary())
