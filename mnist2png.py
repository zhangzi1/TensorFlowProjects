import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
from PIL import Image


# MNIST生成.png
def mnist2png(num, path):
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    batch_image, batch_label = mnist.train.next_batch(num)
    matrix = (batch_image * 255).reshape([-1, 28, 28]).astype(np.int8)
    for i in range(len(matrix)):
        pic = matrix[i].astype(np.int8)
        name = 0
        for a in range(len(batch_label[i])):
            if batch_label[i][a] == 1:
                name = a
        image = Image.fromarray(pic, "L")
        image.save(path + str(i) + "_" + str(name) + ".png")


# 生成图像噪声
def randompng(pic_num, file_path_name):
    for k in range(pic_num):
        pic = (np.random.random([28, 28]) * 255).astype(np.int8)
        image = Image.fromarray(pic, "L")
        image.save(file_path_name)
