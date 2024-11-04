# Functionality to compute token embeddings

import numpy as np

from numpy.typing import NDArray
from typing import Dict

class Embedding():
    """
    Token embedding model.
    
    Args:
        vocabulary_size (int): The number of unique tokens in the corpus
        embedding_dim (int): Dimension of the token vector embedding
    """
    def __init__(self, vocabulary_size: int, embedding_dim: int):
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim

        self.ctx = None # Used to store values for backpropagation

        self.U = None
        self.V = None
        self.reset_parameters()

    def reset_parameters(self):
        """
        We initialize weight matrices U and V of dimension (D, N) and (N, D), respectively
        """
        self.ctx = None
        self.U = np.random.normal(0, np.sqrt(6. / (self.embedding_dim + self.vocabulary_size)), (self.embedding_dim, self.vocabulary_size))
        self.V = np.random.normal(0, np.sqrt(6. / (self.embedding_dim + self.vocabulary_size)), (self.vocabulary_size, self.embedding_dim))

    def one_hot(self, sequence: NDArray, num_classes: int) -> NDArray:
        """
        Given a vector returns a matrix with rows corresponding to one-hot encoding.
        
        Args:
            sequence (NDArray, shape [t]): A sequence of length t containing tokens represented by integers from [0, self.vocabulary_size - 1]
            num_classes (int): How many potential classes (i.e. tokens) there are
            
        Returns:
            NDArray, shape [vocabulary_size, t]: The one-hot encoded representation of `sequence`
        """

        ###########################
        # YOUR CODE HERE
        ###########################
        M = len(sequence)
        one_hot = np.zeros((num_classes, M))
        one_hot[sequence, np.arange(M)] = 1
        return one_hot

    def softmax(self, x: NDArray, axis: int) -> NDArray:
        """
        Computes a numerically stable version of the softmax along an axis.
        
        Args:
            x (NDArray): The input to normalize, any non-empty matrix.
            axis (int): Along which axis to normalize, i.e. along which dimension the softmax is performed.
        
        Returns:
            y (NDArray): Array with same dimension as `input`, but with normalized values.
        """
        
        # Note! You should implement a numerically stable version of softmax
        
        ###########################
        # YOUR CODE HERE
        ###########################
        x_stable = x - np.max(x, axis=axis, keepdims=True)
        y = np.exp(x_stable) / np.sum(np.exp(x_stable), axis=axis, keepdims=True)
        return y

    def loss(self, y_true: NDArray, y_predicted: NDArray) -> float:
        """
        Computes the cross-entropy loss $-1 / M * sum_i(sum_j(y_ij * log(prob_ij)))$ for
        predicted probabilities and ground-truth probabilities. 
        
        Parameters
        ----------
        y: array
            (num_samples, vocabulary_size) matrix of M samples where columns are one-hot vectors for true values
        prob: array
            (num_samples, vocabulary_size) column of M samples where columns are probability vectors after softmax

        Returns
        -------
        loss: float
            Cross-entropy loss calculated as: -1 / M * sum_i(sum_j(y_ij * log(prob_ij)))
        """

        y_predicted = np.clip(y_predicted, 1e-8, None)
        
        ###########################
        # YOUR CODE HERE
        ###########################
        y_predicted = np.log(y_predicted)
        loss = (1/y_true.shape[1]) * -np.sum(y_true*y_predicted)
        return loss

    def forward(self, x: NDArray, y: NDArray) -> float:
        """
        Performs forward pass and saves activations for backward pass
        
        Args:
            x (NDArray, shape [sequence_length], dtype int): Mini-batch of token indices to predict contexts for
            y (NDArray, shape [sequence_length], dtype int): Mini-batch of output context tokens
        
        Returns:
            float: The cross-entropy loss
        """
        
        # Input transformation
        """
        Input is represented with M-dimensional vectors
        We convert them to (vocabulary_size, sequence_length) matrices such that columns are one-hot 
        representations of the input
        """
        x = self.one_hot(x, self.vocabulary_size)
        y = self.one_hot(y, self.vocabulary_size)
        
        # Forward propagation, needs to compute the following
        """
        Returns
        -------
        embedding (NDArray, shape [embedding_dim, sequence_length]): matrix where columns are token embedding from U matrix
        logits (NDArray, shape [vocabulary_size, sequence_length]): matrix where columns are output logits
        prob (NDArray, shape [vocabulary_size, sequence_length]): matrix where columns are output probabilities
        """
        
        ###########################
        # YOUR CODE HERE
        ###########################

        embedding = self.U @ x
        logits = self.V @ embedding
        prob = self.softmax(logits, 0)
        
        # Save values for backpropagation
        self.ctx = (embedding, logits, prob, x, y)
        
        # Loss calculation
        loss = self.loss(y, prob)
        
        return loss
        
    def backward(self) -> Dict[str, NDArray]:
        """
        Given parameters from forward propagation, returns gradient of U and V.
        
        Returns
        -------
        Dict: Gradients with the following keys:
            V (NDArray, shape [vocabulary_size, embedding_dim]) matrix of partial derivatives of loss w.r.t. V
            U (NDArray, shape [embedding_dim, vocabulary_size]) matrix of partial derivatives of loss w.r.t. U
        """

        embedding, logits, prob, x, y = self.ctx

        ###########################
        # YOUR CODE HERE
        ###########################

        dL_dLog = (1/y.shape[1])*(prob-y)
        dLog_dV = embedding.T
        dLog_dEmbed = self.V.T
        dEmbed_dU = x.T

        d_V = dL_dLog @ dLog_dV
        d_U = dLog_dEmbed @ dL_dLog @ dEmbed_dU

        return { 'V': d_V, 'U': d_U }