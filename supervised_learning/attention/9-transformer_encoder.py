#!/usr/bin/env python3
"""
A class Encoder that creates an encoder for a transformer.
"""
import tensorflow as tf


EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock
positional_encoding = __import__('4-positional_encoding').positional_encoding


class Encoder(tf.keras.layers.Layer):
    """A class Encoder that creates an encoder for a transformer."""
    def __init__(
            self,
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_len,
            drop_rate=0.1
    ):
        """
        Initializes an Encoder instance.

        Parameters:
        - N (int): The number of blocks in the encoder.
        - dm (int): The dimensionality of the model.
        - h (int): The number of heads.
        - hidden (int): The number of hidden units in the fully connected layer.
        - input_vocab (int): The size of the input vocabulary.
        - max_seq_len (int): The maximum sequence length possible.
        - drop_rate (float): The dropout rate.

        Sets the following public instance attributes:
        - N: the number of blocks in the encoder
        - dm: the dimensionality of the model
        - embedding: the embedding layer for the inputs
        - positional_encoding: a numpy.ndarray of
         shape (max_seq_len, dm) containing the positional encodings
        - blocks: a list of length N containing all of the EncoderBlock‘s
        - dropout: the dropout layer, to be applied to the positional encodings
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(
                dm,
                h,
                hidden,
                drop_rate
            ) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Performs the forward pass for the encoder.

        Parameters:
        - x: a tensor of shape (batch, input_seq_len, dm)
          containing the input to the encoder
        - training: a boolean to determine if the model is training
        - mask: the mask to be applied for multi head attention

        Returns:
        - A tensor of shape (batch, input_seq_len, dm)
          containing the encoder output.
        """
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding
        x = self.embedding(x)  # (batch_size, input_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocksi

        return x  # (batch_size, input_seq_len, dm)
