#!/usr/bin/env python3
"""
A class SelfAttention that calculates
 the attention for machine translation.
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """A class SelfAttention that calculates
     the attention for machine translation."""
    def __init__(self, units):
        """
        Initializes a SelfAttention instance.

        Parameters:
        - units (int): Number of hidden units
         in the alignment model.

        Sets the following public instance attributes:
        - W: a Dense layer with units units,
         to be applied to the previous decoder hidden state
        - U: a Dense layer with units units,
         to be applied to the encoder hidden states
        - V: a Dense layer with 1 units, to be applied
         to the tanh of the sum of the outputs of W and U
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Performs the forward pass for the attention mechanism.

        Parameters:
        - s_prev: a tensor of shape (batch, units)
         containing the previous decoder hidden state
        - hidden_states: a tensor of shape
         (batch, input_seq_len, units)
         containing the outputs of the encoder

        Returns:
        - context: a tensor of shape (batch, units)
         that contains the context vector for the decoder
        - weights: a tensor of shape (batch, input_seq_len, 1)
         that contains the attention weights
        """
        query_with_time_axis = tf.expand_dims(s_prev, 1)
        score = self.V(
            tf.nn.tanh(
                self.W(query_with_time_axis) + self.U(hidden_states)
            )
        )
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
