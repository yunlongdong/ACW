import sys
sys.path.append('../')

from LayerLib import Dense
from LossLib import L2 
from OpLib import ReLU, Sigmoid
from ACW_utils import summary

import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(100, 64)
y = np.random.randn(100, 1)

D1 = Dense(64, 10)
relu = ReLU()
D2 = Dense(10, 1)
L2Loss = L2()

net = [D1, relu, D2, L2Loss]


lr = 2e-4

for i in range(5000):
    # forward
    y1 = D1(x)
    y2 = relu(y1) 
    y3 = D2(y2)
    loss = L2Loss(y3, y)
    print(i, loss)


    # backward
    grad_loss, _ = L2Loss.backward(y3, y)
    grad_input3, grad_w2 = D2.backward(y2, grad_loss)
    grad_input2, _ = relu.backward(y1, grad_input3)
    grad_input1, grad_w1 = D1.backward(x, grad_input2)

    D1.param -= lr* grad_w1
    D2.param -= lr* grad_w2



summary(net)
plt.plot(y3, 'r-', label='pred')
plt.plot(y, 'g', label='truth')
plt.legend()
plt.show()










