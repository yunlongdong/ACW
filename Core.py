import numpy as np

class Op:
    def __init__(self, name):
        self.name = name
        pass
    
    def __repr__(self):
        return 'Op:' + self.name

    def forward(self, x):
        pass

    def backward(self, grad_y):
        pass
    
    def __call__(self, x):
        return self.forward(x)
        

class Layer:
    def __init__(self, name):
        self.name = name
        self.param = None
        pass
    
    def __repr__(self):
        return 'layer:' + self.name

    def forward(self, x):
        pass

    def backward(self, grad_y):
        pass

    def __call__(self, x):
        return self.forward(x)

class Loss:
    def __init__(self, name):
        self.name = name
        pass
    
    def __repr__(self):
        return 'loss:' + self.name

    def forward(self, x, y):
        pass

    def backward(self, grad_y):
        pass
        
    def __call__(self, x):
        return self.forward(x)
    


