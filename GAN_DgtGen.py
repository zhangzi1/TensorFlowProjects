#
# 基于对抗生成网络的手写数字生成
#


import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from PIL import Image


# Matrix concatenation, 100*784 -> 280*280
def image_concat(input_image_list):
    row_image = input_image_list[0].reshape([28, 28])
    for a in range(1, 10):
        image = input_image_list[a].reshape([28, 28])
        row_image = np.concatenate([row_image, image], axis=1)
    whole_image = row_image

    for i in range(1, 10):
        row_image = input_image_list[i * 10].reshape([28, 28])
        for j in range(1, 10):
            image = input_image_list[i * 10 + j].reshape([28, 28])
            row_image = np.concatenate([row_image, image], axis=1)
        whole_image = np.concatenate([whole_image, row_image], axis=0)
    return (whole_image * 255).astype(np.int8)


# Basic neuron layer
def layer(layer_scope, input_matrix, preceding_layer_neuron_num, this_layer_neuron_num, keep_prob,
          activation_function=None):
    with tf.variable_scope(layer_scope):
        weights = tf.get_variable("Weight", [preceding_layer_neuron_num, this_layer_neuron_num],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("Bias", [this_layer_neuron_num], initializer=tf.constant_initializer(value=0.0))
        y = tf.matmul(input_matrix, weights) + biases
        y = tf.nn.dropout(y, keep_prob)
        if activation_function is None:
            output_matrix = y
        else:
            output_matrix = activation_function(y)
        return output_matrix


# Generator
def gen_network(network_scope, input):
    with tf.variable_scope(network_scope):
        l1 = layer("l1", input, 100, 128, 1.0, tf.nn.relu)
        return layer("output", l1, 128, 28 * 28, 1.0, tf.nn.sigmoid)


# Discriminator
def dis_network(network_scope, input):
    with tf.variable_scope(network_scope):
        l1 = layer("l1", input, 28 * 28, 128, keep_prob, tf.nn.relu)
        l2 = layer("l2", l1, 128, 10, keep_prob, tf.nn.relu)
        return layer("output", l2, 10, 1, 1.0, tf.nn.sigmoid)


# Place Holder
G_random_data = tf.placeholder(tf.float32, shape=[None, 100], name="random_data")
D_input_image = tf.placeholder(tf.float32, shape=[None, 28 * 28], name="MNIST_image")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# Generator
gen_image = gen_network("Gen", G_random_data)

# Discriminator for fake images, shared
fake_output = dis_network("Dis", gen_image)

# Discriminator for real images, shared
with tf.variable_scope("", reuse=True):
    real_output = dis_network("Dis", D_input_image)

# Losses
D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.ones_like(fake_output)))

# Variable lists
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'Dis' in var.name]
g_vars = [var for var in t_vars if 'Gen' in var.name]

# Optimizer
D_training = tf.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
G_training = tf.train.AdamOptimizer().minimize(G_loss, var_list=g_vars)

# Saver
saver = tf.train.Saver()

# Session
sess = tf.Session()

if not os.path.exists("./Model/GAN_DgtGen/"):

    # Initialize
    sess.run(tf.global_variables_initializer())

    # Read data set
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

    # Constant input means constant image
    constant_random_data = np.random.uniform(-1, 1, [100, 100])

    # Training
    for i in range(10000):

        image, _ = mnist.train.next_batch(100)
        sess.run(D_training,
                 feed_dict={D_input_image: image, G_random_data: np.random.uniform(-1, 1, [100, 100]), keep_prob: 0.5})

        sess.run(G_training, feed_dict={G_random_data: np.random.uniform(-1, 1, [100, 100]), keep_prob: 0.5})

        if i == 0 or (i + 1) % 100 == 0:
            # Print losses
            print("\n#" + str(i))
            print(sess.run(D_loss,
                           feed_dict={D_input_image: image, G_random_data: np.random.uniform(-1, 1, [100, 100]),
                                      keep_prob: 1.0}))
            print(sess.run(G_loss, feed_dict={G_random_data: np.random.uniform(-1, 1, [100, 100]), keep_prob: 1.0}))

            # Inspect judgement
            image_p, _ = mnist.train.next_batch(1)
            print(sess.run(real_output, feed_dict={D_input_image: image_p, keep_prob: 1.0}))
            print(sess.run(fake_output, feed_dict={G_random_data: np.random.uniform(-1, 1, [1, 100]), keep_prob: 1.0}))

            # Generate PNG
            output_image = sess.run(gen_image,
                                    feed_dict={G_random_data: constant_random_data, keep_prob: 1.0})
            matrix = image_concat(output_image)
            matimage = Image.fromarray(matrix, "L")
            matimage.save("./Gen_pic/" + time.asctime(time.localtime(time.time())) + " Epoch:" + str(i) + ".png")

    # Save parameters
    saver.save(sess, "./Model/GAN_DgtGen/")

else:

    # Load parameters
    saver.restore(sess, "./Model/GAN_DgtGen/")


