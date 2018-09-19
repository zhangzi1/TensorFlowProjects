#
# 基于循环神经网络的特定序列识别
#


import os
import random

import tensorflow as tf


# 部署一层神经元
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


# 占位
data = tf.placeholder(tf.float32, shape=[1, None, 1])  # 输入层两个神经元
label = tf.placeholder(tf.float32, shape=[1, None, 1])

# 一层循环神经元，5个
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=5)

# 初始状态置0
h0 = cell.zero_state(1, tf.float32)  # 对于每份数据的128个循环神经元初始状态置0

# 一层神经元完成全部循环计算
# inputs: shape = (batch_size, time_steps, input_size)
# cell: RNNCell
# initial_state: shape = (batch_size, cell.state_size)。初始状态。一般可以取零矩阵
outputs, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)  # [1,None,5]

# 一层普通神经元
l1 = nn_layer(outputs[0], 5, 1, 1.0, tf.nn.tanh)

# 损失与优化
loss = tf.reduce_mean(tf.square(label[0] - l1))
training = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 产生训练数据，一半对一半错
train_seq = []
train_rec = []
for i in range(0, 50):
    train_seq.append([[[0], [1], [2], [1], [0]]])
    train_rec.append([[[0], [0], [0], [0], [1]]])
for i in range(0, 50):
    set = [0, 1, 2]
    a0 = random.sample(set, 1)
    a1 = random.sample(set, 1)
    a2 = random.sample(set, 1)
    a3 = random.sample(set, 1)
    a4 = random.sample(set, 1)
    train_seq.append([[a0, a1, a2, a3, a4]])
    train_rec.append([[[0], [0], [0], [0], [0]]])

# 开启模型
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 保存/读取
saver = tf.train.Saver()

if not os.path.exists("./Model/RNN_SeqRec/"):

    # 训练
    for i in range(10000):
        seq = list(range(0, 100))
        a = random.sample(seq, 1)
        sess.run(training, feed_dict={data: train_seq[a[0]], label: train_rec[a[0]]})
        if i % 1000 == 0:
            print(sess.run(loss, feed_dict={data: train_seq[a[0]], label: train_rec[a[0]]}))

    # 保存参数
    saver.save(sess, "./Model/RNN_SeqRec/")  # file_name如果不存在的话，会自动创建

else:

    # 读取保存的参数
    saver.restore(sess, "./Model/RNN_SeqRec/")

# 判别
print(sess.run(l1, feed_dict={data: [[[0], [1], [2], [1], [0]]]}))  # 真例
print(sess.run(l1, feed_dict={data: [[[1], [2], [1], [0], [0]]]}))  # 假例
