import struct
import numpy as np


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


if __name__ == '__main__':
    x = np.array([1, 2, 3])
    y = np.array([[4, 5], [7, 8], [6, 9]])
    print(x.dot(y))