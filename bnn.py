import tensorflow as tf
from tensorflow.keras import layers, constraints, initializers, models
import numpy as np

class Binarizer(layers.Layer):
    def __init__(self, **kwargs):
        super(Binarizer, self).__init__(**kwargs)

    def binarize(self, inputs):
        return tf.where(inputs < 0, -1.0, 1.0)

    def call(self, inputs):
        return self.binarize(inputs)

class DenseLayer(layers.Layer):
    def __init__(self, num_neurons, activation='tanh', use_bias=True, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.num_neurons = num_neurons
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=(input_shape[-1], self.num_neurons),
            initializer=initializers.RandomNormal(stddev=0.1),
            constraint=constraints.MinMaxNorm(-1, 1.0, 1.0, 0),
        )

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=(self.num_neurons,),
                initializer="zeros",
            )
        else:
            self.bias = None

        binarized_kernel = tf.where(self.kernel < 0, -1.0, 1.0)
        self.kernel.assign(binarized_kernel)

        super(DenseLayer, self).build(input_shape)

    def call(self, inputs):
        print('Linear Layer is called...')
        binarizer = Binarizer()
        binary_inputs = binarizer(inputs)
        binary_weights = binarizer(self.kernel)

        binary_inputs = tf.cast(binary_inputs, dtype=tf.float32)
        binary_weights = tf.cast(binary_weights, dtype=tf.float32)

        binary_outputs = tf.matmul(binary_inputs, binary_weights)
        if self.use_bias:
            binary_outputs += self.bias

        if self.activation == 'tanh':
            binary_outputs = tf.math.tanh(binary_outputs)
        # Add more activation functions as needed
        return binary_outputs

class BINLayer(layers.Layer):
    def __init__(self, num_neurons, **kwargs):
        super(BINLayer, self).__init__(**kwargs)
        self.num_neurons = num_neurons

    def binarize(self, inputs):
        return tf.where(inputs < 0, -1, 1)

    def call(self, inputs):
        print('BIN Layer is called...')
        binarized_inputs = self.binarize(inputs)
        return binarized_inputs

def step_activation(x, threshold=0):
    return (x > threshold) * 2 - 1

class Blocks(models.Sequential):
    def __init__(self, num_dense_layer, num_neuron_in_hidden_dense_layer=3, num_neuron_output_layer=3, input_shape=None, training=False):
        super().__init__()
        self.add(layers.InputLayer(input_shape=input_shape))

        for i in range(num_dense_layer):
            activation = 'tanh' if i < num_dense_layer - 1 else step_activation
            self.add(DenseLayer(num_neurons=num_neuron_in_hidden_dense_layer, activation=activation))
        # Uncomment the line below if you want a BINLayer after the last DenseLayer
        # self.add(BINLayer(num_neurons=num_neuron_output_layer))

class BNN(tf.keras.Model):
    def __init__(self, num_dense_layer=3, num_neuron_in_dense_layer=3, num_neuron_output_layer=6, input_shape=(10, 10)):
        super().__init__()
        self.num_dense_layer = num_dense_layer
        self.num_neuron_in_dense_layer = num_neuron_in_dense_layer
        self.num_neuron_output_layer = num_neuron_output_layer
        self.blocks = Blocks(num_dense_layer=num_dense_layer,
                             num_neuron_in_hidden_dense_layer=num_neuron_in_dense_layer,
                             num_neuron_output_layer=num_neuron_output_layer,
                             input_shape=input_shape)

    def call(self, inputs):
        x = inputs
        for layer in self.blocks.layers:
            x = layer(x)

        return x

