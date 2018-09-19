#
# 基于梯度下降优化的线性回归
#


import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if __name__ == '__main__':

    # 输入量占位
    x = tf.placeholder(tf.float32, [1, 1])
    label = tf.placeholder(tf.float32, [1, 1])

    # 线性模型
    w = tf.Variable(tf.random_normal([1, 1]))
    b = tf.Variable(tf.random_normal([1, 1]))
    y = tf.add(tf.matmul(w, x), b)

    # 计算、优化损失
    loss = tf.square(label - y)
    training = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # 制造训练数据
    training_x = np.array([])
    training_y = np.array([])
    for i in range(20):
        training_x = np.append(training_x, i)
        training_y = np.append(training_y, 2 * i + 3 + np.random.normal(loc=0.0, scale=1, size=None))

    # 建立会话
    sess = tf.Session()

    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 保存/读取
    saver = tf.train.Saver()

    if not os.path.exists("./Model/LinearRegression/"):

        # 训练若干次
        for i in range(10000):
            seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            a = random.sample(seq, 1)  # 随机序号
            sess.run(training, feed_dict={x: [[training_x[a[0]]]], label: [[training_y[a[0]]]]})
            if i % 100 == 0:
                print(sess.run(loss, feed_dict={x: [[training_x[a[0]]]], label: [[training_y[a[0]]]]}))

        # 保存参数
        saver.save(sess, "./Model/LinearRegression/")

    else:

        # 读取保存的参数
        saver.restore(sess, "./Model/LinearRegression/")

    # 拟合量
    print("\nw:", sess.run(w))
    print("\nb:", sess.run(b))

    # 制图
    reg_y = sess.run(w)[0][0] * training_x + sess.run(b)[0][0]
    plt.scatter(training_x, training_y)
    plt.plot(training_x, reg_y)
    plt.show()

    sess.close()
