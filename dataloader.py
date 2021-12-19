import struct
import numpy as np
import dataset
import os


def idx1_loader(filename):
    bin_data = open(filename, 'rb').read()
    offset = 0
    fmt = '>ii'
    magic_number, size = struct.unpack_from(fmt, bin_data, offset)

    labels = np.empty(size, int)

    offset += struct.calcsize(fmt)
    fmt = '>B'

    for i in range(size):
        labels[i] = struct.unpack_from(fmt, bin_data, offset)[0]
        offset += struct.calcsize(fmt)

    y = np.zeros((size, 10))
    for i in range(size):
        y[i][labels[i]] = 1

    return y


def idx3_loader(filename):
    bin_data = open(filename, 'rb').read()
    offset = 0
    fmt = '>iiii'
    magic_number, image_number, row, col = struct.unpack_from(fmt, bin_data, offset)

    offset += struct.calcsize(fmt)
    fmt = '>' + str(row * col) + 'B'

    images = np.empty((image_number, row * col))

    for i in range(image_number):
        images[i] = struct.unpack_from(fmt, bin_data, offset)
        offset += struct.calcsize(fmt)

    t = np.ones(image_number)

    x = np.insert(images, 784, t, 1)

    return x / 256.0


def load_data():
    cur_path = os.path.abspath(os.curdir)
    train_set = dataset.Dataset(
            idx3_loader(cur_path + r'\手写数据集\train-images.idx3-ubyte'),
            idx1_loader(cur_path + r'\手写数据集\train-labels.idx1-ubyte')
        )
    tx = idx3_loader(cur_path + r'\手写数据集\t10k-images.idx3-ubyte')
    ty = idx1_loader(cur_path + r'\手写数据集\t10k-labels.idx1-ubyte')
    test_set = dataset.Dataset(tx[:2500], ty[:2500])
    cross_set = dataset.Dataset(tx[2500:], ty[2500:])
    return train_set, test_set, cross_set


if __name__ == '__main__':
    a, b, c = load_data()
