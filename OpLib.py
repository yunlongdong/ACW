from Core import Op 
import numpy as np

    
class ReLU(Op):

    def __init__(self):
        super().__init__('Op:ReLU')

    def forward(self, x):
        return (x>0) * x

    def backward(self, x, grad_y):
        grad_y[x<0] = 0
        return grad_y, 0


class Sigmoid(Op):

    def __init__(self):
        super().__init__('Op:Sigmoid')

    def forward(self, x):
        return 1/(1 + np.exp(-x))

    def backward(self, x, grad_y):
        grad = self.forward(x) * (1 - self.forward(x))
        return grad*grad_y, 0
