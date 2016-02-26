# -*- coding: utf-8 -*-
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

import numpy as np
from matplotlib import pyplot as plt
import math


def read_mnist(img_src, label_src, n):
    with open(img_src, 'r') as f:
        magic, n_imgs, n_rows, n_cols = np.fromfile(f, '>I', 4)
        imgs = np.fromfile(f, 'u1', n_imgs*n_rows*n_cols).reshape(n_imgs, n_rows*n_cols)
    with open(label_src, 'r') as f:
        magic, n_labels, = np.fromfile(f, '>I', 2)
        labels = np.fromfile(f, 'u1', n_labels)
    return imgs[:n], labels[:n]


def mnist_read_train_data(img_src='../datasets/mnist/train-images.idx3-ubyte',
                          label_src='../datasets/mnist/train-labels.idx1-ubyte', n=60000):
    return read_mnist(img_src, label_src, n)


def mnist_read_test_data(img_src='../datasets/mnist/t10k-images.idx3-ubyte',
                         label_src='../datasets/mnist/t10k-labels.idx1-ubyte', n=10000):
    return read_mnist(img_src, label_src, n)


def mnist_show_images(images):
    images_per_row = math.ceil(np.sqrt(len(images)))
    fig = plt.figure()
    for i in range(len(images)):
        ax = fig.add_subplot(images_per_row, images_per_row, i+1)
        ax.imshow(images[i].reshape(28, 28), cmap='gray_r')
        plt.axis('off')
    plt.show()
