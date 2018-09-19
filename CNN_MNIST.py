#
# 基于卷积神经网络的的28*28手写数字识别
#


import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from PIL import Image


# 添加层
def layer(input_matrix, preceding_layer_neuron_num, this_layer_neuron_num, keep_prob, activation_function=None):
    weights = tf.Variable(tf.truncated_normal(shape=[preceding_layer_neuron_num, this_layer_neuron_num], stddev=0.1),
                          name="Weight")
    biases = tf.Variable(tf.constant(0.1, shape=[this_layer_neuron_num]), name="Bias")
    y = tf.matmul(input_matrix, weights) + biases
    # 防止过拟合
    y = tf.nn.dropout(y, keep_prob)
    if activation_function is None:
        output_matrix = y
    else:
        output_matrix = activation_function(y)
    return output_matrix


# CNN的层组
def conv_grp(input_image, kernel_shape):
    # Convolution
    kernel = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), tf.float32, name="Kernel")
    bias = tf.Variable(tf.constant(0.1, shape=[kernel_shape[-1]]), name="Bias")
    image_1 = tf.nn.conv2d(input_image, kernel, [1, 1, 1, 1], "SAME") + bias
    # ReLU
    image_2 = tf.nn.relu(image_1)
    # Pooling，输入图像(长宽+1)/2
    image_3 = tf.nn.max_pool(image_2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
    return image_3


# 1*784的矩阵识别
def digit_recognition(image_path_or_matrix):
    # 重制图
    tf.reset_default_graph()
    # 占位
    feature = tf.placeholder(tf.float32, [None, 28 * 28])
    label = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # feature转为图像矩阵
    image = tf.reshape(feature, [-1, 28, 28, 1])

    # 卷积组
    cg1 = conv_grp(image, [5, 5, 1, 32])
    cg2 = conv_grp(cg1, [5, 5, 32, 64])

    # 展平
    vote = tf.reshape(cg2, [-1, 7 * 7 * 64])

    # 全连接层
    fc1 = layer(vote, 7 * 7 * 64, 1024, keep_prob, tf.nn.relu)
    fc2 = layer(fc1, 1024, 10, 1.0, tf.nn.softmax)

    # 误差与优化
    cross_entropy = -tf.reduce_sum(label * tf.log(fc2))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 准确率
    correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(label, 1))
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 保存/读取
    saver = tf.train.Saver()

    # 开启模型
    sess = tf.Session()

    if not os.path.exists("./Model/CNN_MNIST/"):

        # 初始化变量
        sess.run(tf.global_variables_initializer())

        # 调取数据
        mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

        # 开始训练
        start_time = time.time()
        for i in range(1000):
            batch_x, batch_y = mnist.train.next_batch(100)  # batch
            if i % 100 == 0:
                print("#", str(i), "  Batch accuracy:",
                      sess.run(train_accuracy, feed_dict={feature: batch_x, label: batch_y, keep_prob: 1.0}))
            sess.run(train_step, feed_dict={feature: batch_x, label: batch_y, keep_prob: 0.5})
        end_time = time.time()

        print("Time taken:", round(end_time - start_time, 4), "s")

        # 训练集准确率
        print("MNIST accuracy:",
              sess.run(train_accuracy,
                       feed_dict={feature: mnist.test.images, label: mnist.test.labels, keep_prob: 1.0}))

        # 保存参数
        saver.save(sess, "./Model/CNN_MNIST/")

    else:

        # 读取保存的参数
        saver.restore(sess, "./Model/CNN_MNIST/")

    # 识别
    result = None
    if isinstance(image_path_or_matrix, str):

        # png -> matrix
        im = Image.open(image_path_or_matrix)
        image = np.reshape(list(im.getdata()), [-1, 28 * 28]) / 255.0

        result = sess.run(tf.argmax(sess.run(fc2, feed_dict={feature: image, keep_prob: 1.0}), 1))
    elif isinstance(image_path_or_matrix, list) or isinstance(image_path_or_matrix, np.ndarray):
        result = sess.run(tf.argmax(sess.run(fc2, feed_dict={feature: image_path_or_matrix, keep_prob: 1.0}), 1))

    # 关闭会话
    sess.close()

    # 返回结果
    return result


if __name__ == '__main__':
    im = Image.open("./selfmade_pic/7.png")
    image = np.reshape(list(im.getdata()), [-1, 28 * 28]) / 255.0
    print(digit_recognition(image))

    print(digit_recognition("./selfmade_pic/9.png"))
