import numpy as np
import pdb
from scipy.signal import convolve2d as conv2d

class Conv2D:
    def __init__(self, kernel_size=3, stride=1):
        self.kernel_size = kernel_size
        self.stride = stride
        #self.K = np.random.randn(self.kernel_size, self.kernel_size)
        self.K = np.eye(3)
        self.K[0, 2] = 1
        self.pad_size = int((self.kernel_size -1)/2)
    
    def forward(self, X):
        self.X = X
        self.input_width = X.shape[1]
        self.input_height = X.shape[0]
        # causion , using scipy convolve2d, need to transpose the kernel
        return conv2d(self.X, self.K.T, mode='valid', boundary='fill', fillvalue=0)

    def im2col(self, im):
        # padding image
        pad_size = self.pad_size
        padim = np.pad(im, pad_size, 'constant', constant_values = 0)

        # width and height
        Width = padim.shape[1]
        Height = padim.shape[0]

        # num of subimage to convolve
        num_conv = ((Height - self.kernel_size)/self.stride + 1) * ((Width - self.kernel_size)/self.stride + 1)
        num_conv = int(num_conv)
        colim = np.zeros((self.kernel_size * self.kernel_size, num_conv))

        # get subimage of image to convolve
        index = 0
        for i in range(0, Height - self.kernel_size + 1, self.stride):
            for j in range(0, Width - self.kernel_size + 1, self.stride):
                subim = padim[i:(i+self.kernel_size), j:(j+self.kernel_size)]
                # expand in row-major order
                subim = subim.flatten('C')
                colim[:, index] = subim
                index += 1
        return colim
    
    def col2im(self, col_matrix):
        im = np.zeros((2*self.pad_size + self.X.shape[0], 2*self.pad_size + self.X.shape[1]))
        Width = im.shape[1]
        pad_size = self.pad_size
        for index in range(col_matrix.shape[1]):
            row = index / ((Width - self.kernel_size)/self.stride + 1)
            row = int(row)
            col = index - row*((Width - self.kernel_size)/self.stride + 1)
            col = int(col)
            sub_im  = col_matrix[:, index].reshape((self.kernel_size, self.kernel_size))
            index += 1
            im[row:(row+self.kernel_size), col:(col+self.kernel_size)] = sub_im
        return im

    def k2col(self):    
        return self.K.flatten('C').reshape((1, self.kernel_size**2))

    def backward(self, grad_output):
        col_grad_output = grad_output.flatten('C').reshape((1, -1))
        # the gradient of input, Y = k * X, * stands for convolv, grad_X = grad_Y.T * k.
        # reference in https://zhuanlan.zhihu.com/p/41392664
        # causion , using scipy convolve2d, need to transpose the kernel
        grad_input = conv2d(self.K, grad_output.T.T, mode='valid', boundary='fill', fillvalue=0)
        col_grad_K = col_grad_output.dot(self.im2col(self.X).T)
        grad_K = col_grad_K.reshape(self.K.shape)
        pdb.set_trace()
        return grad_input, grad_K
        
if __name__ == '__main__':
    X = np.array([[1, 2], [3, 4]])
    conv = Conv2D()
    convX = conv.forward(X)
    colX = conv.im2col(X)
    im = conv.col2im(colX)
    
    grad_input, grad_K = conv.backward(convX)



