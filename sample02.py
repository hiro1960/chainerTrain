#!/usr/bin/env python
# coding: utf-8

import random
import argparse
import numpy
import chainer
import chainer.optimizers

#
# XORの学習プログラム　１層では線形しか学習できないので、２層にする
#

class ClassificationModel(chainer.FunctionSet):
    def __init__(self):
        super(ClassificationModel, self).__init__(
            # ニューラルネットを２層にする
            fc1 = chainer.functions.Linear(2, 2),
            fc2 = chainer.functions.Linear(2, 2)
            )
    def _forward(self, x):
        h = self.fc2(chainer.functions.sigmoid(self.fc1(x)))
        return h
        
    def train(self, x_data, y_data):
        x = chainer.Variable(x_data.reshape(1,2).astype(numpy.float32), volatile=False)
        y = chainer.Variable(y_data.astype(numpy.int32), volatile=False)
        h = self._forward(x)

        optimizer.zero_grads()
        error = chainer.functions.softmax_cross_entropy(h, y)
        error.backward()
        # 例題ではoptimizerが抜けていたが、これがないと学習がうまくいかない
        optimizer.update()


        print("x: {}".format(x.data))
        print("h: {}".format(h.data))
        print("h_class: {}".format(h.data.argmax()))

# プログラム本体

model = ClassificationModel()
optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model.collect_parameters())


# XORの学習データ

data_xor = [
    [numpy.array([0,0]), numpy.array([0])],
    [numpy.array([0,1]), numpy.array([1])],
    [numpy.array([1,0]), numpy.array([1])],
    [numpy.array([1,1]), numpy.array([0])],
]*2000

for invec, outvec in data_xor:
    model.train(invec, outvec)