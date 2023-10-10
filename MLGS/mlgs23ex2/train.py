# Functionality to train the token embedding module

import numpy as np
from numpy.typing import NDArray
from typing import Dict

class Optimizer():
    """
    Stochastic gradient descent with momentum optimizer.

    Parameters:
        model (Embedding): The embedding module to train
        learning_rate (float): The learning rate
        momentum (float, optional, default: 0.0): Momentum for gradient computation
    
    ----------
    model: object
        Model as defined above
    learning_rate: float
        Learning rate
    momentum: float (optional)
        Momentum factor (default: 0)
    """
    def __init__(self, model: object, learning_rate: float, momentum: float=0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.previous = None # Previous gradients
    
    def _init_previous(self, grad: Dict[str, NDArray]):
        # Initialize previous gradients to zero
        self.previous = { k: np.zeros_like(v) for k,v in grad.items() }
    
    def step(self, grad: Dict[str, NDArray]):
        if self.previous is None:
            self._init_previous(grad)
            
        for name, dw in grad.items():
            dw_prev = self.previous[name]
            w = getattr(self.model, name)

            """
            Given weight w, previous gradients dw_prev and current 
            gradients dw, performs an update of weight w.

            Computes:
                dw_new (NDArray): New gradients calculated as combination of previous and
                    current, weighted with momentum factor.
                w_new (NDArray):
                    New weights calculated with a single step of gradient
                    descent using dw_new.
            """
            ###########################
            # YOUR CODE HERE
            ###########################

            dw_new = self.momentum*dw_prev + (1-self.momentum) * dw
            w_new = w - self.learning_rate*dw_new

            self.previous[name] = dw_new
            setattr(self.model, name, w_new)