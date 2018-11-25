# ACW: a light-weight deep learning framework based on numpy

It is still improving. The main design of this framework it to simplify the structure. I devide all stuff common in deep learning models into 3 categories:

* Layer (LayerLib.py): perform some transform on input and have parameters to be optimize (such as Dense, Conv, and so on). The api is following Keras somewhat.
* Op (OpLib.py): operation on input but with no parameters, such as add, multiply, maxpool, non-linear activation and so on.
* Loss (LossLib.py): The loss function used to minimize. such as MSE, L1Loss, CrossEntropy and so on.


### example

A MLP example has been provide in ```example/``` dir.



