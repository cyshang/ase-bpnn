"""Defines neural network layers in BPNN

Each class represents a fully connected layer
"""

import tensorflow as tf


class relu:
    def __init__(self, name, n_in, n_out, parameters=None):
        self.n_in = n_in
        self.n_out = n_out
        if not parameters:
            self.parameters = {}
            self.parameters['W'] = tf.get_variable(name + '-W',
                                                   [n_in, n_out],
                                                   initializer=tf.contrib.
                                                   layers.xavier_initializer())
            self.parameters['b'] = tf.get_variable(name + '-b',
                                                   [1, n_out],
                                                   initializer=tf.contrib.
                                                   layers.xavier_initializer())

    def __repr__(self):
        return 'Relu(%i, %i)' % (self.n_in, self.n_out)

    def __call__(self, nn_input):
        return tf.nn.relu(tf.matmul(nn_input, self.parameters['W']) +
                          self.parameters['b'])


class linear:
    def __init__(self, name, n_in, n_out, parameters=None):
        self.n_in = n_in
        self.n_out = n_out
        if not parameters:
            self.parameters = {}
            self.parameters['W'] = tf.get_variable(name + '-W', [n_in, n_out],
                                                   initializer=tf.contrib.
                                                   layers.xavier_initializer())
            self.parameters['b'] = tf.get_variable(name + '-b', [1, n_out],
                                                   initializer=tf.contrib.
                                                   layers.xavier_initializer())

    def __repr__(self):
        return 'Linear(%i, %i)' % (self.n_in, self.n_out)

    def __call__(self, nn_input):
        return tf.matmul(nn_input, self.parameters['W']) + self.parameters['b']
