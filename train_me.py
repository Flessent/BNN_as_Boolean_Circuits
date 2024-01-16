import tensorflow as tf
from tensorflow.keras import layers, constraints, initializers,models
from math import ceil
import numpy as np
import sys
from itertools import chain
import time
from cnf_to_bdd import * 


class Binarizer(layers.Layer):
    def __init__(self, **kwargs):
        super(Binarizer, self).__init__(**kwargs)

    def binarize(self, inputs):
        return tf.where(inputs < 0, -1.0, 1.0)

    def call(self, inputs):
        return self.binarize(inputs)
class BinaryInitializer:
    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):
        return tf.constant(np.random.choice([-1, 1], size=shape), dtype=dtype)

class DenseLayer(layers.Layer):
    def __init__(self, num_neurons, activation='tanh', use_bias=True, **kwargs):
        super(DenseLayer, self).__init__( **kwargs)
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
        output = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            output = tf.add(output, self.bias)
        if self.activation == 'tanh':
            output = tf.math.tanh(output)
        # Add more activation functions as needed
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_neurons)


    def call(self, inputs):
        print('Linear Layer is called...')
        binarizer = Binarizer()
        binary_inputs = binarizer(inputs)
        binary_weights = binarizer(self.kernel)

        # Explicitly cast binary_inputs and binary_weights to float32
        binary_inputs = tf.cast(binary_inputs, dtype=tf.float32)
        binary_weights = tf.cast(binary_weights, dtype=tf.float32)

        binary_outputs = tf.matmul(binary_inputs, binary_weights) + self.bias
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



class Blocks(models.Sequential):
    def __init__(self, num_dense_layer, num_neuron_in_hidden_dense_layer=3,  num_neuron_output_layer=3, input_shape=None, training=False):
        super().__init__()
        self.add(layers.InputLayer(input_shape=input_shape))

        for _ in range(num_dense_layer):
            self.add(DenseLayer(num_neurons=num_neuron_in_hidden_dense_layer))
        self.add(DenseLayer(num_neurons=num_neuron_output_layer))
          

class BNN(tf.keras.Model):
    def __init__(self, num_dense_layer=3, num_neuron_in_dense_layer=3, num_neuron_output_layer=6,input_shape=(10,10)):
        super().__init__()
        self.num_dense_layer=num_dense_layer
        self.num_neuron_in_dense_layer=num_neuron_in_dense_layer
        self.num_neuron_output_layer=num_neuron_output_layer
        self.blocks = Blocks(num_dense_layer=num_dense_layer, num_neuron_in_hidden_dense_layer=num_neuron_in_dense_layer,num_neuron_output_layer=num_neuron_output_layer,input_shape=input_shape)


    def call(self, inputs):
        x = inputs
        # Internal Blocks
        for layer in self.num_internal_blocks:
            x = layer(x)

        # Output Blocks
        for layer in self.num_output_blocks:
            x = layer(x)

        return x



if __name__ == "__main__":
     bnn_model = BNN(num_dense_layer=2,num_neuron_in_dense_layer=10,num_neuron_output_layer=4)
     dimacs_file_path=encode_network(bnn_model)
     #describe_network(bnn_model)

     #dimacs_file_path = 'output_final.cnf'
     n_vars = 10 
    # CNF to BDD
     cnf_formula = read_dimacs_file(dimacs_file_path)
     output_file_path = 'output_bdd_info.txt'

     bdd_compiler = BDD_Compiler(n_vars, cnf_formula)
     bdd = bdd_compiler.compile(output_file=output_file_path)
    # Optional: Visualize the BDD
     bdd.print_info(n_vars)





