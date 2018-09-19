#
# 基于深度神经网络的的异或运算
#


import os
import random

import tensorflow as tf


# 添加层，设置【数据正向传播】的相关计算
def nn_layer(input_matrix, preceding_layer_neuron_num, this_layer_neuron_num, keep_prob, activation_function=None):
    weights = tf.Variable(tf.truncated_normal(shape=[preceding_layer_neuron_num, this_layer_neuron_num], stddev=0.5),
                          name="Weight")
    biases = tf.Variable(tf.constant(0.5, shape=[this_layer_neuron_num]), name="Bias")
    y = tf.matmul(input_matrix, weights) + biases
    y = tf.nn.dropout(y, keep_prob)
    if activation_function is None:
        output_matrix = y
    else:
        output_matrix = activation_function(y)
    return output_matrix


if __name__ == "__main__":

    # 设置feature和label占位符
    data = tf.placeholder(tf.float32, [1, 2], name="feature")
    label = tf.placeholder(tf.float32, [1, 1], name="label")
    keep_prob = tf.placeholder(tf.float32)

    # 设置隐含层和输出层，首尾相接体现【数据正向传播】
    l1 = nn_layer(data, 2, 10, keep_prob, activation_function=tf.nn.tanh)
    l2 = nn_layer(l1, 10, 10, keep_prob, activation_function=tf.nn.tanh)
    l3 = nn_layer(l2, 10, 10, keep_prob, activation_function=tf.nn.tanh)
    l4 = nn_layer(l3, 10, 10, keep_prob, activation_function=tf.nn.tanh)
    l5 = nn_layer(l4, 10, 10, keep_prob, activation_function=tf.nn.tanh)
    l6 = nn_layer(l5, 10, 10, keep_prob, activation_function=tf.nn.tanh)
    l7 = nn_layer(l6, 10, 1, 1.0, activation_function=tf.nn.tanh)

    # 计算损失，并使用梯度下降法来修正参数，学习率0.1
    loss = tf.reduce_mean(tf.square(label - l7))
    training = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 训练数据集
    training_data = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
    training_label = [[[0]], [[1]], [[1]], [[0]]]

    # 建立会话，激活模型
    sess = tf.Session()

    # 初始化所有tf.Variable
    init = tf.global_variables_initializer()
    sess.run(init)

    # 保存/读取
    saver = tf.train.Saver()

    if not os.path.exists("./Model/DNN_XOR/"):

        # 训练模型
        for i in range(10000):
            seq = [0, 1, 2, 3]
            a = random.sample(seq, 1)  # 随机序号
            sess.run(training, feed_dict={data: training_data[a[0]], label: training_label[a[0]], keep_prob: 1.0})
            if i % 1000 == 0:
                print("#", str(i), "Updated error:",
                      sess.run(loss,
                               feed_dict={data: training_data[a[0]], label: training_label[a[0]], keep_prob: 1.0}))

        # 保存参数
        saver.save(sess, "./Model/DNN_XOR/")  # file_name如果不存在的话，会自动创建

    else:

        # 读取保存的参数
        saver.restore(sess, "./Model/DNN_XOR/")

    # 使用模型进行预测并计时
    print(sess.run(l7, feed_dict={data: [[1, 0]], keep_prob: 1.0}))

    # 关闭会话，关闭模型
    sess.close()
