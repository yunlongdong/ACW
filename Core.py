import numpy as np

class Op:
    def __init__(self, name):
        self.name = name
        pass
    
    def __repr__(self):
        return 'Op:' + self.name

    def foward(self, x):
        pass

    def backward(self, grad_y):
        pass
        

class Layer:
    def __init__(self, name):
        self.name = name
        self.param = None
        pass
    
    def __repr__(self):
        return 'layer:' + self.name

    def foward(self, x):
        pass

    def backward(self, grad_y):
        pass

class Loss:
    def __init__(self, name):
        self.name = name
        pass
    
    def __repr__(self):
        return 'loss:' + self.name

    def foward(self, x, y):
        pass

    def backward(self, grad_y):
        pass
        
    


