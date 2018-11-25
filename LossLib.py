from Core import Loss 
import numpy as np

    
class L2(Loss):

    def __init__(self):
        super().__init__('L2')

    def forward(self, y_pred, y):
        return 1/2 * np.linalg.norm(y_pred - y, 2)

    def __call__(self, y_pred, y):
        return self.forward(y_pred, y)

    def backward(self, y_pred, y):
        return y_pred - y, 0


