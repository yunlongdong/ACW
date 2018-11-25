from LayerLib import Dense
from LossLib import L2 
import numpy as np
from utils import summary

x = np.random.randn(4, 2)
y = np.random.randn(4, 1)

D1 = Dense(2, 10)
D2 = Dense(10, 10)
D3 = Dense(10, 1)
L2Loss = L2()

net = [D1, D2, D3, L2Loss]
summary(net)

# forward
y_pred1 = D1.forward(x)
y_pred2 = D2.forward(y_pred1)
y_pred3 = D3.forward(y_pred2)
loss = L2Loss.forward(y_pred2, y)

# backward
grad_y_pred, _ = L2Loss.backward(y_pred3, y)
grad_input3, grad_dense3_param = D3.backward(y_pred2, grad_y_pred)
grad_input2, grad_dense2_param = D2.backward(y_pred1, grad_input3)
grad_input1, grad_dense1_param = D1.backward(x, grad_input2)











