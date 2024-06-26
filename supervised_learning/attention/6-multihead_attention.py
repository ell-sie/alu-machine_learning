#!/usr/bin/env python3
"""
A class MultiHeadAttention that performs multi head attention.
"""
import tensorflow as tf


sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """A class MultiHeadAttention that performs multi head attention."""
    def __init__(self, dm, h):
        """
        Initializes a MultiHeadAttention instance.

        Parameters:
        - dm (int): Dimensionality of the model.
        - h (int): Number of heads.

        Sets the following public instance attributes:
        - h: the number of heads
        - dm: the dimensionality of the model
        - depth: the depth of each attention head
        - Wq: a Dense layer with dm units, used to generate the query matrix
        - Wk: a Dense layer with dm units, used to generate the key matrix
        - Wv: a Dense layer with dm units, used to generate the value matrix
        - linear: a Dense layer with dm units,
         used to generate the attention output
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Splits the last dimension of tensor x into (h, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Performs the forward pass for the multi head attention mechanism.

        Parameters:
        - Q: a tensor of shape (batch, seq_len_q, dk)
         containing the input to generate the query matrix
        - K: a tensor of shape (batch, seq_len_v, dk)
         containing the input to generate the key matrix
        - V: a tensor of shape (batch, seq_len_v, dv)
         containing the input to generate the value matrix
        - mask: always None

        Returns:
        - output: a tensor with its last two dimensions
         as (..., seq_len_q, dm) containing the scaled dot product attention
        - weights: a tensor with its last three dimensions
         as (..., h, seq_len_q, seq_len_v) containing the attention weights
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)  # (batch_size, seq_len, dm)
        K = self.Wk(K)  # (batch_size, seq_len, dm)
        V = self.Wv(V)  # (batch_size, seq_len, dm)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scaled_attention, attention_weights = sdp_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        output = self.linear(concat_attention)  # (batch_size, seq_len_q, dm)

        return output, attention_weights
