#!/usr/bin/env python3
"""
Defines the class RNNCell that represents a cell of a simple RNN
"""


import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN

    class constructor:
        def __init__(self, i, h, o)

    public instance attributes:
        Wh: concatenated hidden state and input data weights
        bh: concatenated hidden state and input data biases
        Wy: output weights
        by: output biases

    public instance methods:
        def forward(self, h_prev, x_t):
            performs forward propagation for one time step
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        parameters:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs

        creates public instance attributes:
            Wh: concatenated hidden state and input data weights
            bh: concatenated hidden state and input data biases
            Wy: output weights
            by: output biases

        weights should be initialized using random normal distribution
        weights will be used on the right side for matrix multiplication
        biases should be initiliazed as zeros
        """
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def softmax(self, x):
        """
        Performs the softmax function

        parameters:
            x: the value to perform softmax on to generate output of cell

        return:
            softmax of x
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)
        return softmax

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        parameters:
            h_prev [numpy.ndarray of shape (m, h)]:
                contains previous hidden state
                m: the batch size for the data
                h: dimensionality of hidden state
            x_t [numpy.ndarray of shape (m, i)]:
                contains data input for the cell
                m: the batch size for the data
                i: dimensionality of the data

        output of the cell should use softmax activation function

        returns:
            h_next, y:
            h_next: the next hidden state
            y: the output of the cell
        """
        concatenation = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concatenation, self.Wh) + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
