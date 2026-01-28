import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid of x.

    Arguments:
    x -- A scalar, python list, or numpy array of any size.

    Return:
    s -- sigmoid(x) as a numpy array.

    """
    # 1. Convert input to a numpy array (handles scalars, lists, and arrays uniformly)
    x_arr = np.array(x, dtype=float)
    
    # 2. Compute sigmoid using the formula: 1 / (1 + e^-x)
    # np.exp works element-wise on arrays (vectorization)
    s = 1 / (1 + np.exp(-x_arr))
    return s
    
    pass