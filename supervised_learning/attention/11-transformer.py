#!/usr/bin/env python3
"""
A class Transformer that creates a transformer network.
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder

class Transformer(tf.keras.Model):
    """A class Transformer that creates a transformer network."""
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Initializes a Transformer instance.

        Parameters:
        - N (int): The number of blocks in the encoder and decoder.
        - dm (int): The dimensionality of the model.
        - h (int): The number of heads.
        - hidden (int): The number of hidden units in the fully connected layers.
        - input_vocab (int): The size of the input vocabulary.
        - target_vocab (int): The size of the target vocabulary.
        - max_seq_input (int): The maximum sequence length possible for the input.
        - max_seq_target (int): The maximum sequence length possible for the target.
        - drop_rate (float): The dropout rate.

        Sets the following public instance attributes:
        - encoder: the encoder layer
        - decoder: the decoder layer
        - linear: a final Dense layer with target_vocab units
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
        """
        Performs the forward pass for the transformer.

        Parameters:
        - inputs: a tensor of shape (batch, input_seq_len) containing the inputs
        - target: a tensor of shape (batch, target_seq_len) containing the target
        - training: a boolean to determine if the model is training
        - encoder_mask: the padding mask to be applied to the encoder
        - look_ahead_mask: the look ahead mask to be applied to the decoder
        - decoder_mask: the padding mask to be applied to the decoder

        Returns:
        - A tensor of shape (batch, target_seq_len, target_vocab) containing the transformer output.
        """
        enc_output = self.encoder(inputs, training, encoder_mask)  # (batch, inp_seq_len, dm)

        # dec_output.shape == (batch, tar_seq_len, dm)
        dec_output, attention_weights = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask)

        final_output = self.linear(dec_output)  # (batch, tar_seq_len, target_vocab)

        return final_output, attention_weights
