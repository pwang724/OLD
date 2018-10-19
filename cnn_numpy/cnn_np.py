'''
Practice building a convolutional network from numpy.

'''

import numpy as np
import Data.loadData as load_data
from scipy import signal as sg
import matplotlib.pyplot as plt


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1.0 - y)

def reLU(x):
    return np.maximum(0, x)

def dreLU(x):
    return x > 0


class ConvNet(object):
    def __init__(self, layers, x, y):
        nL = len(layers)
        weight = np.empty(nL, dtype=object)
        bias = np.empty(nL, dtype=object)
        a = np.empty(nL, dtype=object)
        d = np.empty(nL, dtype=object)
        a[0] = np.empty(1, dtype=object)  # input
        inputmap = 1
        mapsize = x[1, ::].shape
        mapsize = np.array(mapsize)

        for l in range(nL):
            type = layers[l]['Type']
            if type == 'POOL':
                m = float(mapsize[0]) / layers[l]['Scale']
                assert m.is_integer()
                mapsize = mapsize / layers[l]['Scale']
                a[l] = np.empty(inputmap, dtype=object)
                d[l] = np.empty(inputmap, dtype=object)

            elif type == 'CONV':
                kernelSize = layers[l]['KernelSize']
                outputmap = layers[l]['OutputMap']
                mapsize = mapsize - kernelSize + 1
                fan_in = float(inputmap * kernelSize ** 2)
                fan_out = float(layers[l]['OutputMap'] * kernelSize ** 2)

                kCur = np.empty((inputmap, outputmap), dtype=object)
                for j in range(inputmap):
                    for k in range(outputmap):
                        kCur[j][k] =.01 * (np.random.rand(kernelSize, kernelSize) - .5) * \
                                     2 * np.sqrt(6 / (fan_in + fan_out))
                weight[l] = kCur
                bias[l] = np.zeros(outputmap)
                a[l] = np.empty(outputmap, dtype=object)
                d[l] = np.empty(outputmap, dtype=object)
                inputmap = outputmap

            elif type == 'FF':
                inp = int(np.prod(mapsize) * inputmap)
                if layers[l]['Size']:
                    outp = layers[l]['Size']
                elif l == nL - 1:
                    outp = y[0, :].size
                else:
                    outp = inp
                bias[l] = np.zeros((int(outp), 1))
                w = np.random.rand(outp, inp)
                # weight[l] = .01 * (w - .5) * 2 * np.sqrt(6 / float(inp + outp))
                weight[l] = (w - .5) * 2 / (inp + outp)

            elif type == 'FLATTEN':
                d[l] = np.empty(inputmap, dtype=object)
                print('')

            elif type == 'INPUT':
                print('')

            else:
                print('Layer type not recognized')

        self.nL = nL
        self.weight = weight
        self.bias = bias
        self.layers = layers
        self.a = a
        self.d = d
        self.dw = np.empty(nL, dtype=object)
        self.db = np.empty(nL, dtype=object)

    def ff(self, x):
        self.a[0][0] = x
        inputmaps = 1

        for l in range(1, self.nL):
            type = self.layers[l]['Type']
            if type == 'CONV':
                imSize = np.asarray(self.a[l - 1][0].shape)
                kSize = self.layers[l]['KernelSize']
                dims = imSize - [0, kSize - 1, kSize - 1]
                outputmaps = self.layers[l]['OutputMap']

                for j in range(outputmaps):
                    z = np.zeros(dims)
                    for i in range(inputmaps):
                        k = self.weight[l][i][j]
                        k = k[np.newaxis, ::-1, ::-1]  # flip k
                        z = z + sg.convolve(self.a[l - 1][i], k, mode='valid')

                    self.a[l][j] = reLU(z + self.bias[l][j])
                inputmaps = outputmaps

            elif type == 'POOL':
                scale = self.layers[l]['Scale']
                filter = np.ones((1, scale, scale)) / scale ** 2
                for j in range(inputmaps):
                    z = sg.convolve(self.a[l - 1][j], filter, 'valid')
                    self.a[l][j] = z[:, ::scale, ::scale]

            elif type == 'FLATTEN':
                nS = self.a[l - 1][0].shape[0]
                fv = np.empty((nS, 0))
                for i in range(self.a[l - 1].size):
                    curA = self.a[l - 1][i]
                    t = curA.reshape(curA.shape[0], curA.shape[1] * curA.shape[2])
                    fv = np.hstack((t, fv))
                self.a[l] = fv

            elif type == 'FF':
                nS = self.a[l - 1].shape[0]
                fv = self.a[l - 1]
                b = np.tile(self.bias[l].T, (nS, 1))
                a = np.dot(fv, self.weight[l].T) + b

                if l == self.nL - 1:
                    self.a[l] = sigmoid(a)
                else:
                    self.a[l] = reLU(a)

    def bp(self, y):
        yhat = self.a[-1]
        error = yhat - y
        self.d[-1] = dsigmoid(yhat) * error

        for l in reversed(range(self.nL - 1)):
            type = self.layers[l]['Type']
            if type == 'FF':
                d = np.dot(self.weight[l + 1].T, self.d[l + 1].T)
                self.d[l] = dreLU(self.a[l]) * d.T

            elif type == 'FLATTEN':  ##unflatten
                d = np.dot(self.d[l + 1], self.weight[l + 1])
                sa = self.a[l - 1][0].shape
                chunk = np.product(sa[1:])
                for j in range(self.a[l - 1].size):
                    s = j * chunk
                    e = (j + 1) * chunk
                    self.d[l][j] = d[:, s:e].reshape(-1, sa[1], sa[2])

            elif type == 'POOL':
                if self.layers[l + 1]['Type'] == 'FLATTEN':
                    self.d[l] = self.d[l + 1]

                elif self.layers[l + 1]['Type'] == 'CONV':
                    inputmap = self.a[l].size
                    outputmap = self.a[l + 1].size
                    for i in range(inputmap):
                        z = np.zeros(self.a[l][0].shape)
                        for j in range(outputmap):
                            k = self.weight[l + 1][i][j]
                            k = k[np.newaxis, :, :]
                            z = z + sg.convolve(self.d[l + 1][j], k, mode='full')
                        self.d[l][i] = z

            elif type == 'CONV':
                if self.layers[l + 1]['Type'] == 'FLATTEN':
                    for j in range(self.d[l].size):
                        self.d[l][j] = self.d[l + 1][j] * dreLU(self.a[l][j])

                if self.layers[l + 1]['Type'] == 'POOL':
                    for j in range(self.a[l].size):
                        pooled = self.d[l + 1][j]
                        scale = self.layers[l + 1]['Scale']
                        p = np.repeat(pooled, scale, axis=1)
                        p = np.repeat(p, scale, axis=2)
                        p = np.divide(p, scale ** 2)
                        self.d[l][j] = p * dreLU(self.a[l][j])

    def calcgrad(self):
        dw = np.empty(self.nL, dtype=object)
        db = np.empty(self.nL, dtype=object)
        for l in range(1, self.nL):
            type = self.layers[l]['Type']
            if type == 'FF':
                dw[l] = np.dot(self.d[l].T, self.a[l - 1]) / self.d[l].shape[0]
                db[l] = np.mean(self.d[l].T, axis=1).reshape(-1, 1)

            if type == 'CONV':
                dw[l] = np.empty(self.weight[l].shape, dtype=object)
                db[l] = np.empty(self.bias[l].shape)
                for i in range(self.a[l - 1].size):
                    for j in range(self.a[l].size):
                        o = self.a[l - 1][i]
                        d = self.d[l][j]
                        dw[l][i][j] = np.squeeze(sg.convolve(o, d[::-1, ::-1, ::-1], mode='valid')) / d.shape[0]
                        db[l][j] = d.sum() / d.shape[0]
        self.dw = dw
        self.db = db

    def applygrads(self, opts):
        a = opts['alpha']
        for l in range(1, self.nL):
            type = self.layers[l]['Type']
            if type == 'FF':
                self.weight[l] = self.weight[l] - (a * self.dw[l])
                self.bias[l] = self.bias[l] - (a * self.db[l])

            if type == 'CONV':
                for j in range(self.a[l].size):
                    for i in range(self.a[l - 1].size):
                        self.weight[l][i][j] = self.weight[l][i][j] - (a * self.dw[l][i][j])
                    self.bias[l][j] = self.bias[l][j] - (a * self.db[l][j])

    def train(self, x, y, opts):
        epochs = opts['numepochs']
        nS = x.shape[0]
        batch_size = opts['batch_size']
        num_batches = np.floor(nS / batch_size)
        rem = nS % batch_size

        for ep in range(epochs):
            kk = np.random.permutation(nS)
            if rem != 0:
                kk = kk[:-rem]

            for l in range(int(num_batches-1)):
                s = l * batch_size
                e = (l + 1) * batch_size
                batch_x = x[kk[s:e], :, :]
                batch_y = y[kk[s:e], :]

                NN.ff(batch_x)
                NN.bp(batch_y)
                NN.calcgrad()
                NN.applygrads(opts)
            # plt.imshow(NN.a[-1])
            # plt.pause(.1)
            if ep % 20 == 0:
                err = NN.test(x,y)
                print('[*] Epoch %d  err=%.3f' % (ep, err))

    def test(self, x, y):
        self.ff(x)
        yhat = self.a[-1]
        yhat = np.argmax(yhat, axis=1)
        y = np.argmax(y, axis=1)
        err = np.equal(yhat, y).mean()
        return err

# # MAKE INPUT
# tr_X, tr_Y, ts_X, ts_Y = load_data.mnist()
# tr_X = tr_X.reshape(-1, 28, 28)
# ts_X = ts_X.reshape(-1, 28, 28)
# INPUT = {'Type': 'INPUT'}
# LC0 = {'Type': 'CONV', 'KernelSize': 5, 'OutputMap': 5}
# LP0 = {'Type': 'POOL', 'Scale': 2}
# LC1 = {'Type': 'CONV', 'KernelSize': 5, 'OutputMap': 10}
# LP1 = {'Type': 'POOL', 'Scale': 2}
# LF0 = {'Type': 'FLATTEN'}
# LF1 = {'Type': 'FF', 'Size': []}
# layers = (INPUT, LC0, LP0, LF0, LF1)

tr_X, tr_Y, ts_X, ts_Y = load_data.digits('./Data/sklearn_digits.csv')
tr_X = tr_X.reshape(-1, 8, 8)
ts_X = ts_X.reshape(-1, 8, 8)
INPUT = {'Type': 'INPUT'}
LC0 = {'Type': 'CONV', 'KernelSize': 3, 'OutputMap': 5}
LP0 = {'Type': 'POOL', 'Scale': 2}
LF0 = {'Type': 'FLATTEN'}
LF1 = {'Type': 'FF', 'Size': []}
layers = (INPUT, LC0, LP0, LF0, LF1)

# TRAINING OPTIONS
opts = {'alpha': .001, 'batchsize': 50, 'numepochs': 200}
NN = ConvNet(layers, tr_X, tr_Y)
NN.train(tr_X, tr_Y, opts)
err = NN.test(ts_X, ts_Y)
print('[!] test err=%.3f' % (err))