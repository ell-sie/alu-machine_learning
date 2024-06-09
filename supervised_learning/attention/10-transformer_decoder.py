#!/usr/bin/env python3
"""
A class Decoder that creates a decoder for a transformer.
"""
import tensorflow as tf
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock
positional_encoding = __import__('4-positional_encoding').positional_encoding

class Decoder(tf.keras.layers.Layer):
    """A class Decoder that creates a decoder for a transformer."""
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        """
        Initializes a Decoder instance.

        Parameters:
        - N (int): The number of blocks in the encoder.
        - dm (int): The dimensionality of the model.
        - h (int): The number of heads.
        - hidden (int): The number of hidden units in the fully connected layer.
        - target_vocab (int): The size of the target vocabulary.
        - max_seq_len (int): The maximum sequence length possible.
        - drop_rate (float): The dropout rate.

        Sets the following public instance attributes:
        - N: the number of blocks in the encoder
        - dm: the dimensionality of the model
        - embedding: the embedding layer for the targets
        - positional_encoding: a numpy.ndarray of shape (max_seq_len, dm) containing the positional encodings
        - blocks: a list of length N containing all of the DecoderBlock‘s
        - dropout: the dropout layer, to be applied to the positional encodings
        """
        super(Decoder, self).__init__()
        self.dm = dm
        self.N = N
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Performs the forward pass for the decoder.

        Parameters:
        - x: a tensor of shape (batch, target_seq_len, dm) containing the input to the decoder
        - encoder_output: a tensor of shape (batch, input_seq_len, dm) containing the output of the encoder
        - training: a boolean to determine if the model is training
        - look_ahead_mask: the mask to be applied to the first multi head attention layer
        - padding_mask: the mask to be applied to the second multi head attention layer

        Returns:
        - A tensor of shape (batch, target_seq_len, dm) containing the decoder output.
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x, block1, block2 = self.blocksi

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
