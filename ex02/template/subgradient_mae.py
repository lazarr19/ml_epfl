import numpy as np

def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    e = y - np.ravel(np.dot(tx, w))
    
    def subgradient_abs(x):
        if x>0: 
            return 1 
        elif x<0:
            return -1
        else:
            return 0.1
    
    subgradient_abs = np.vectorize(subgradient_abs)    
    return np.dot(subgradient_abs(e), -tx)/e.shape[0]