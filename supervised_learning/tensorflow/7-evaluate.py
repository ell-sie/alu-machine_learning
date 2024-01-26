#!/usr/bin/env python3
"""
Module for evaluationg a neural network classifier
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network
    """
    saver = tf.train.import_meta_graph(save_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        prediction = sess.run(y_pred, feed_dict={'x:0': X, 'y:0': Y})
        cost = sess.run(loss, feed_dict={'x:0': X, 'y:0': Y})
        acc = sess.run(accuracy, feed_dict={'x:0': X, 'y:0': Y})
    return prediction, acc, cost
