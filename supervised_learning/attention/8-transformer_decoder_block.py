#!/usr/bin/env python3
"""
A class DecoderBlock that creates an encoder block for a transformer.
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention

class DecoderBlock(tf.keras.layers.Layer):
    """A class DecoderBlock that creates an encoder block for a transformer."""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes a DecoderBlock instance.

        Parameters:
        - dm (int): Dimensionality of the model.
        - h (int): Number of heads.
        - hidden (int): Number of hidden units in the fully connected layer.
        - drop_rate (float): Dropout rate.

        Sets the following public instance attributes:
        - mha1: the first MultiHeadAttention layer
        - mha2: the second MultiHeadAttention layer
        - dense_hidden: the hidden dense layer with hidden units and relu activation
        - dense_output: the output dense layer with dm units
        - layernorm1: the first layer norm layer, with epsilon=1e-6
        - layernorm2: the second layer norm layer, with epsilon=1e-6
        - layernorm3: the third layer norm layer, with epsilon=1e-6
        - dropout1: the first dropout layer
        - dropout2: the second dropout layer
        - dropout3: the third dropout layer
        """
        super(DecoderBlock, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Performs the forward pass for the decoder block.

        Parameters:
        - x: a tensor of shape (batch, target_seq_len, dm) containing the input to the decoder block
        - encoder_output: a tensor of shape (batch, input_seq_len, dm) containing the output of the encoder
        - training: a boolean to determine if the model is training
        - look_ahead_mask: the mask to be applied to the first multi head attention layer
        - padding_mask: the mask to be applied to the second multi head attention layer

        Returns:
        - A tensor of shape (batch, target_seq_len, dm) containing the block’s output.
        """
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch, target_seq_len, dm)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)  # (batch, target_seq_len, dm)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch, target_seq_len, dm)

        ffn_output = self.dense_hidden(out2)  # (batch, target_seq_len, hidden)
        ffn_output = self.dense_output(ffn_output)  # (batch, target_seq_len, dm)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch, target_seq_len, dm)

        return out3
