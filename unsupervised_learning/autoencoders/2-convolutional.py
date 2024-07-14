#!/usr/bin/env python3
"""
    Creates an autoencoder
"""
import numpy as np
import tensorflow.keras as keras

# Set floating-point precision for numpy
np.set_printoptions(precision=8)

def autoencoder(input_dims, filters, latent_dims):
    """
    Creates an autoencoder
    :param input_dims: an integer containing the dimensions of model input
    :param filters: a list containing the number of filters for each
    convolutional layer in the encoder and decoder, respectively
    :param latent_dims: an integer containing the dimensions of the latent
    space representation
    :return: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
    input_encoder = keras.Input(shape=input_dims)
    input_decoder = keras.Input(shape=latent_dims)

    # Encoder model
    encoded = input_encoder
    for filter in filters:
        encoded = keras.layers.Conv2D(filters=filter, kernel_size=(3, 3), padding='same', activation='relu')(encoded)
        encoded = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)

    # Latent layer
    encoder = keras.Model(inputs=input_encoder, outputs=encoded)

    # Decoder model
    decoded = input_decoder
    for filter in filters[::-1]:
        decoded = keras.layers.Conv2D(filters=filter, kernel_size=(3, 3), padding='same', activation='relu')(decoded)
        decoded = keras.layers.UpSampling2D(size=(2, 2))(decoded)

    decoded = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3), padding='same', activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=input_decoder, outputs=decoded)

    # Autoencoder model
    auto_input = keras.Input(shape=input_dims)
    encoded_auto = encoder(auto_input)
    decoded_auto = decoder(encoded_auto)
    auto = keras.Model(inputs=auto_input, outputs=decoded_auto)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

# Example usage
input_dims = (28, 28, 1)  # Example input dimensions
filters = [32, 64, 128]   # Example filter sizes
latent_dims = (4, 4, 128)  # Example latent dimensions

encoder, decoder, auto = autoencoder(input_dims, filters, latent_dims)

# Check the autoencoder structure
auto.summary()
