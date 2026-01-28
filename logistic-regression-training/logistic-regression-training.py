import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    # This prevents overflow for large negative values of z
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # 1. Convert inputs to numpy arrays to handle lists/arrays uniformly
    X = np.array(X)
    y = np.array(y)
    
    # 2. Get dimensions: N = number of samples, D = number of features
    N, D = X.shape
    
    # 3. Initialize parameters
    # Weights (w) as zeros vector of size D
    # Bias (b) as float 0.0
    w = np.zeros(D)
    b = 0.0
    
    # 4. Gradient Descent Loop
    for _ in range(steps):
        # Forward pass: z = Xw + b
        z = np.dot(X, w) + b
        
        # Activation: p = sigmoid(z)
        p = _sigmoid(z)
        
        # Calculate gradients
        # The gradient of Loss w.r.t w is: (1/N) * X^T * (p - y)
        grad_w = (1 / N) * np.dot(X.T, (p - y))
        
        # The gradient of Loss w.r.t b is: mean(p - y)
        grad_b = np.mean(p - y)
        
        # Update parameters
        w = w - lr * grad_w
        b = b - lr * grad_b
        
    return w, b

    pass