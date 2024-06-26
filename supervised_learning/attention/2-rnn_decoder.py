#!/usr/bin/env python3
"""
A class RNNDecoder that decodes for machine translation.
"""
import tensorflow as tf


SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """A class RNNDecoder that decodes for machine translation."""
    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes a RNNDecoder instance.

        Parameters:
        - vocab (int): Size of the output vocabulary.
        - embedding (int): Dimensionality of the embedding vector.
        - units (int): Number of hidden units in the RNN cell.
        - batch (int): Batch size.

        Sets the following public instance attributes:
        - embedding: a keras Embedding layer that
         converts words from the vocabulary into an embedding vector
        - gru: a keras GRU layer with units units
        - F: a Dense layer with vocab units
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Performs the forward pass for the decoder.

        Parameters:
        - x: a tensor of shape (batch, 1) containing
         the previous word in the target sequence
         as an index of the target vocabulary
        - s_prev: a tensor of shape (batch, units)
          containing the previous decoder hidden state
        - hidden_states: a tensor of shape (batch,
         input_seq_len, units) containing the outputs of the encoder

        Returns:
        - y: a tensor of shape (batch, vocab) containing
         the output word as a one hot vector in the target vocabulary
        - s: a tensor of shape (batch, units) containing
         the new decoder hidden state
        """
        context, weights = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)

        return y, state
