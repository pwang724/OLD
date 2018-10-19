import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets.mldata import fetch_mldata
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


def digits(file):
    data = np.loadtxt(file, delimiter=',')
    # first ten values are the one hot encoded y (target) values
    y = data[:, 0:10]
    data = data[:, 10:]  # x data
    # scale the data so values are between 0 and 1
    data -= data.min()
    data /= data.max()

    train_X, test_X, train_Y, test_Y = train_test_split(data, y, test_size=1 /7.0, random_state=0)
    return train_X, train_Y, test_X, test_Y

def mnist():
    mnist = fetch_mldata('MNIST original', data_home= './mldata')
    onehot = np.zeros((mnist.target.size, 10))
    onehot[np.arange(onehot.shape[0]), mnist.target.astype(int)] = 1

    train_X, test_X, train_Y, test_Y = train_test_split(
        mnist.data, onehot, test_size=1 /7.0, random_state=0)
    train_X = train_X.astype(float) / 255
    test_X = test_X.astype(float) / 255

    # plt.figure(figsize=(20, 4))
    # for index, (image, label) in enumerate(zip(train_X[0:5], train_Y[0:5])):
    #     plt.subplot(1, 5, index + 1)
    #     plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    #     plt.axis('off')

    return train_X, train_Y, test_X, test_Y

if __name__ == "__main__":
    mnist()
    print('done')