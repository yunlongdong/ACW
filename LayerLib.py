from Core import Layer
import numpy as np

class Dense(Layer):

    def __init__(self, n_in, n_out):
        super().__init__('dense_{}_{}'.format(n_in, n_out))
        self.n_in = n_in
        self.n_out = n_out
        self.param =  np.random.randn(n_in, n_out)

    def forward(self, x):
        self.x = x
        w = self.param
        # the dense layer math: y = x*w
        y = x.dot(w)
        return y

    def backward(self, grad_y):
        x = self.x
        return grad_y.dot(self.param.T), x.T.dot(grad_y)
    

class Conv2D(Layer):
    
    def __init__(self, C_in, C_out, K_s, Stride):
        """
        Params:
            in channels, out channles, kernel size, stride
        """
        super().__init__("conv_{}_{}x{}x{}".format(C_out, C_in, K_s, K_s))
        self.c_in = C_in
        self.c_out = C_out
        self.k_s = K_s
        self.stride = Stride
        self.pad_size = int((K_s -1)/2)
        self.num_param = C_in*C_out*K_s*K_s
            
        self.K = np.arange(self.num_param).reshape((C_out, C_in, K_s, K_s))
        
    def k2col(self):
        """
            k2col on kernel, to size C_out*(K_s^2*C_in)
        """
        self.colK =  self.K.flatten('C').reshape((self.c_out, -1))
        return self.colK

    #def im2col(self, im):
    #    """
    #        im2col on im with shape N*C*H*W
    #        and col is of shape (K_s^2*C_in)*?, ? to be determined
    #    """
    #    # only pad on H and W
    #    im_pad = np.pad(im, ((0, 0), (0, 0), (self.pad_size, self.pad_size), (self.pad_size, self.pad_size)), 'constant', constant_values=0)
    #    H = im_pad.shape[2]
    #    W = im_pad.shape[3]
    #    C = im_pad.shape[1]
    #    N = im_pad.shape[0]

    #    # num of patch to  convolve along height and weight
    #    num_patch_h = (H-self.k_s)/self.stride + 1
    #    num_patch_w = (W-self.k_s)/self.stride + 1
    #    # the number of patch to convolve per channel
    #    num_patch_per_sample = num_patch_h * num_patch_w
    #    num_patch = int(num_patch_per_sample * N) 
    #    self.num_patch = num_patch

    #    col = np.zeros((self.k_s**2*C, num_patch))
    #    
    #    # to col
    #    index = 0
    #    for i in range(0, H-self.k_s+1, self.stride):
    #        for j in range(0, W-self.k_s+1, self.stride):
    #            # patch shape N*C*K_s*K_s
    #            patch = im_pad[:, :, i:(i+self.k_s), j:(j+self.k_s)]
    #            patch = patch.flatten('C').reshape((N, -1))
    #            col[:, (index*N):((index+1)*N)] = patch.T
    #            index += 1
    #    return col

    def im2col_g(self, im):
        """
            im2col on im with shape N*C*H*W
            and col is of shape (K_s^2*C_in)*?, ? to be determined
        """
        k_s = self.k_s
        stride = self.stride
        pad_size = int((k_s-1)/2)
        # only pad on H and W
        im_pad = np.pad(im, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=0)
        self.im_pad = im_pad
        H = im_pad.shape[2]
        W = im_pad.shape[3]
        C = im_pad.shape[1]
        N = im_pad.shape[0]

        # num of patch to  convolve along height and weight
        num_patch_h = (H-k_s)/stride + 1
        num_patch_w = (W-k_s)/stride + 1
        # the number of patch to convolve per channel
        num_patch_per_sample = int(num_patch_h * num_patch_w)
        num_patch = int(num_patch_per_sample * N) 

        self.num_patch = num_patch
        self.num_patch_h= num_patch_h
        self.num_patch_w= num_patch_w
        self.num_patch_per_sample = num_patch_per_sample

        col = np.zeros((k_s**2*C, num_patch))
        
        # to col
        for n in range(N):
            index = 0
            for i in range(0, H-k_s+1, stride):
                for j in range(0, W-k_s+1, stride):
                    # patch shape N*C*K_s*K_s
                    patch = im_pad[n, :, i:(i+k_s), j:(j+k_s)]
                    patch = patch.flatten('C').reshape((-1, 1))
                    col[:, int(n*num_patch_per_sample+index)] = patch.T
                    index += 1
        return col

    def col2im_g(self, col):
        """
            col2im on col of shape (K_s^2*C_in)*(num_patch_per_sample*N)
        """
        k_s = int(self.k_s)
        pad_size = int((k_s-1)/2)
        stride = self.stride
        num_patch_per_sample = int(self.num_patch_per_sample)

        im = np.zeros(self.im_pad.shape)
        N = im.shape[0]
        C = im.shape[1]
        H = im.shape[2]
        W = im.shape[3]

        for i in range(N):
            for j in range(self.num_patch_per_sample):
                one_col = col[:, i*num_patch_per_sample + j]
                one_col = one_col.reshape((C, k_s, k_s))
                row_index = int(j//self.num_patch_w)
                col_index = int(j - self.num_patch_w * row_index)
                #print("row:{}, col:{}".format(row_index, col_index))
                # shape N*C*H*W
                im[i, :, row_index:(row_index+k_s), col_index:(col_index+k_s)] = one_col
                
        # remove the padding
        return im[:, :, pad_size:-pad_size, pad_size:-pad_size]
        

    def forward(self, X):
        """
            forward, X is of shape N*C*H*W
        """
        N = X.shape[0]
        C = X.shape[1]
        H = X.shape[2]
        W = X.shape[3]

        colK = self.k2col()
        colX = self.im2col_g(X)
        colY = colK.dot(colX)

        Y = colY.reshape((N, self.c_out, H, W))
    

        # saving for backward calc
        self.X = X
        self.Y = Y
        self.colY = colY
        self.colX = colX

        return Y 

    def any2col(self, A):
        """
            A is of shape N*C*H*W
        """
        N = A.shape[0]
        C = A.shape[1]
        H = A.shape[2]
        W = A.shape[3]
        col = np.zeros((N, C*H*W))
        for i in range(N):
            col[i, :] = A[i, :, :, :].flatten('C').reshape((1, -1))
        return col
        
    
    def backward(self, grad_output):
        col_grad_output = grad_output.reshape(self.colY.shape)

        col_grad_K = col_grad_output.dot(self.colX.T)
        grad_K = col_grad_K.reshape(self.K.shape)
        
        col_grad_input = col_grad_K.T.dot(col_grad_output)
        
        grad_input = self.col2im_g(col_grad_input)
        
        return grad_input, grad_K



