#!/usr/bin/env python3
"""
A class RNNEncoder that defines a single neuron
performing the role of an encoder in a machine translation model.
"""
import tensorflow as tf

class RNNEncoder(tf.keras.layers.Layer):
    """A class RNNEncoder that defines a single neuron
    performing the role of an encoder in a machine translation model."""
    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes a RNNEncoder instance.

        Parameters:
        - vocab (int): Size of the input vocabulary.
        - embedding (int): Dimensionality of the embedding vector.
        - units (int): Number of hidden units in the RNN cell.
        - batch (int): Batch size.

        Sets the following public instance attributes:
        - batch: the batch size
        - units: the number of hidden units in the RNN cell
        - embedding: a keras Embedding layer that
        converts words from the vocabulary into an embedding vector
        - gru: a keras GRU layer with units units
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for
         the RNN cell to a tensor of zeros.

        Returns:
        - A tensor of shape (batch, units)
         containing the initialized hidden states.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Performs the forward pass for the encoder.

        Parameters:
        - x: a tensor of shape (batch, input_seq_len)
         containing the input to the encoder
         layer as word indices within the vocabulary.
        - initial: a tensor of shape (batch, units)
         containing the initial hidden state.

        Returns:
        - outputs: a tensor of shape (batch, input_seq_len, units)
         containing the outputs of the encoder.
        - hidden: a tensor of shape (batch, units)
         containing the last hidden state of the encoder.
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state = initial)
        return outputs, hidden
