#!/usr/bin/env python
# coding:utf-8
import numpy as np
import chainer.functions as F
from chainer import Variable, FunctionSet, optimizers

#
# 先のsampleで回帰がうまく動かなかったので、別の例を探した。
# こちらはうまく動く
#

def init():
    # model definition
    h1_num = 2  # １層あたりのユニット数

    # l1の第１引数で、２ニューロンあることを示している
    model = FunctionSet(
    l1 = F.Linear(2, h1_num),
    l2 = F.Linear(h1_num, 1)
    )

    return model

def forward(x):
    # estimation by model
    h1 = model.l1(x)
    y  = model.l2(F.sigmoid(h1))
    return y

def train(x_data, y_data):
    optimizer.zero_grads()

    # extract input and output
    x = Variable(np.array([in_vec]).astype(np.float32))
    t = Variable(np.array([out_vec]).astype(np.float32))

    y = forward(x)

    # error correction
    loss = F.mean_squared_error(y, t)
#    loss = F.softmax_cross_entropy(y, t)

    # feedback and learning
    loss.backward()
    optimizer.update()

    # print
    if lp % 101 == 0:
        print(lp, in_vec, y.data, out_vec) 
        string = "%d, %d, %d, %f, %d\n" % (lp, in_vec[0], in_vec[1], y.data[0], out_vec)
        fout.write(string)

# プログラム本体

model = init() # モデル生成
optimizer = optimizers.SGD()
#optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9) # こっちでもいけた。引数の影響は不明
optimizer = optimizers.MomentumSGD() # こっちでもいけた。引数の影響は不明
optimizer.setup(model)

# number of learning
times = 1000

# input and output vector
xor_data = [
    [np.array([0,0]), np.array([0])],
    [np.array([0,1]), np.array([1])],
    [np.array([1,0]), np.array([1])],
    [np.array([1,1]), np.array([0])],
] * times

# main routine
fout = open('output.csv', 'w')
lp = 0
for in_vec, out_vec in xor_data:
    train(in_vec, out_vec) # 訓練
    
    # back to top
    lp += 1

fout.close()