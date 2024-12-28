#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024 - 2024 Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam
# @Email  : sepinetam@gmail.com
# @File   : BPNN.py

# BPNN: BP神经网络
"""
TODO:
"""

import numpy as np
import pandas as pd


class BPNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重矩阵和偏置向量
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

        # 激活函数及其导数
        self.activation = lambda x: 1 / (1 + np.exp(-x))
        self.activation_derivative = lambda x: x * (1 - x)

    def forward(self, X):
        # 前向传播
        self.z2 = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.a2 = self.activation(self.z2)

        self.z3 = np.dot(self.a2, self.weights_hidden_output) + self.bias_output
        self.a3 = self.activation(self.z3)

        return self.a3

    def backward(self, X, y, learning_rate):
        # 反向传播
        output_error = y - self.a3
        output_delta = output_error * self.activation_derivative(self.a3)

        hidden_error = np.dot(self.a2.T, output_delta)
        hidden_delta = hidden_error * self.activation_derivative(self.a2)

        # 更新权重和偏置
        self.weights_hidden_output += np.dot(self.a2.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                self.forward(X[i].reshape(1, -1))
                self.backward(X[i].reshape(1, -1), y[i].reshape(1, -1), learning_rate)
            print(f'Epoch {epoch + 1}, Loss: {np.mean(np.square(y - self.a3))}')


if __name__ == '__main__':
    # 使用pandas加载数据
    data_path = "../../../src/dta/cat_dog.csv"
    df = pd.read_csv(data_path)

    X = df.drop('target', axis=1).values  # 特征值
    y = df['target'].values.reshape(-1, 1)  # 目标值

    # 创建BPNN实例
    bpnn = BPNN(input_size=X.shape[1], hidden_size=4, output_size=1)

    # 训练网络
    bpnn.train(X, y, epochs=1000, learning_rate=0.1)
