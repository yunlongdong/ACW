from Core import Layer
import numpy as np

class Dense(Layer):

    def __init__(self, n_in, n_out):
        super().__init__('dense_{}_{}'.format(n_in, n_out))
        self.n_in = n_in
        self.n_out = n_out
        self.param =  np.random.randn(n_in, n_out)

    def forward(self, x):
        w = self.param
        # the dense layer math: y = x*w
        y = x.dot(w)
        return y

    def backward(self, x, grad_y):
        return grad_y.dot(self.param.T), x.T.dot(grad_y)
    




