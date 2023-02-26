import copy
from enum import Enum

import numpy as np

class NeuralNetwork:
    """
    Class representing a neural network.
    """
    
    def __init__(self, layer_node_amounts, activation_functions):
        """
        Initializes the neural network. Layer node amounts must be a tuple 
        containing integers representing the number of nodes in each layer.
        Activation function must be a tuple containing instances of the 
        ActivationFunction class only, it must have a length of one less than 
        the number of layers.
        Example values vector for layer 0 with 3 nodes:
        l0 = [
            n0_0
            n1_0
            n2_0
            1
        ]
        Example weights to layer 1 with 2 nodes:
        w1 = [
            w00_1 w01_1 w02_1 b0_1
            w10_1 w11_1 w12_1 b1_1
            0     0     0     1
        ]
        (wjk_l = $w^l_{jk}$, bn_l = $b_n^l$)
        """
        # Check types.
        func_name = "NeuralNetwork.__init__()"
        
        # Check if layer_node_amounts is a tuple.
        assert isinstance(layer_node_amounts, tuple), f"{func_name}: " \
            "layer_node_amounts must be a tuple containing integers."
        
        # Check if layer_node_amounts only contains integers.
        for n in layer_node_amounts:
            assert isinstance(n, int), f"{func_name} layer_node_amounts " \
                "must be a tuple containing integers."
        
        # Check if activation_functions is a tuple.
        assert isinstance(activation_functions, tuple), f"{func_name}: " \
            "activation_functions must be a tuple containing instances of " \
            "the ActivationFunction enum."
        
        # Check if the length of activation_functions is one less than the 
        # number of layers.
        assert len(activation_functions) == len(layer_node_amounts) - 1, \
            f"{func_name}: length of activation_functions must be equal to " \
            "the number of non-input layers."
        
        # Check if activation_functions contains only instances of 
        # ActivationFunction.
        for a in activation_functions:
            assert isinstance(a, ActivationFunction), f"{func_name}: " \
                "activation_functions must be a tuple containing instances " \
                "of the ActivationFunction enum."
        
        # Set member variables.
        self.layer_node_amounts = layer_node_amounts
        self.activation_functions = (None, ) + activation_functions
        
        # Initialize layer values and weights lists.
        self.values = []
        self.weights = []
        for index, n in enumerate(layer_node_amounts):
            # Add numpy array of the size of the layer.
            values = np.zeros(n + 1, dtype=float)
            values[-1] = 1
            self.values.append(values)
            
            # If this is the first layer, add None as weights.
            if index == 0:
                weights = None
            else:
                weights = np.zeros((layer_node_amounts[index] + 1, \
                    layer_node_amounts[index - 1] + 1), dtype=float)
                weights[-1][-1] = 1
            
            # Append weights matrix to weights list.
            self.weights.append(weights)
            
            # Maybe useful comment:
            # weights[1] * values[0] = values[1]
            # n1+1 x n0+1 * n0+1 x 1 = n1+1 x 1
            # n0+1 for bias node in prev layer
            # n1+1 for adding a 1 to the output vector
        
        # Randomize weights and biases (the last layer should stay all zeros 
        # and a one at the end).
        for w in self.weights:
            # Skip first weights entry.
            if w is None:
                continue
            
            # Generate array of random weights and biases.
            new_weights = np.random.rand(w.shape[0] - 1, w.shape[1]) * 2 - 1
            
            # Replace values in weights list.
            w[0:new_weights.shape[0]] = new_weights
    
    def feed_forward(self, inputs):
        """
        Sets the input layer values to the supplied inputs, feeds them 
        through the network, and returns the outputs.
        """
        # Check types.
        func_name = "NeuralNetwork.feed_forward()"
        
        # Check if inputs is a tuple.
        assert isinstance(inputs, tuple), f"{func_name}: Inputs must be a " \
            "tuple of floats."
        
        # Check if inputs only contains floats.
        for f in inputs:
            assert isinstance(f, float), f"{func_name}: Inputs must be a " \
                "tuple of floats."
        
        # Check if the size of the input vector is correct.
        assert len(inputs) == self.layer_node_amounts[0], f"{func_name}: " \
            f"Inputs must have a length of {self.layer_node_amounts[0]} " \
            "elements."
        
        # Set input values.
        self.values[0][0:self.layer_node_amounts[0]] = inputs
        
        # Loop over all layers.
        for index in range(len(self.values)):
            # Skip weights before the first layer as they do not exist.
            if index == 0:
                continue
            
            # Multiply the weights of the current layer with the values of 
            # the previous layer.
            self.values[index] = np.matmul(self.weights[index], \
                self.values[index - 1])
            
            # Apply activation function.
            match self.activation_functions[index]:
                case ActivationFunction.Linear:
                    self.values[index][:-1] = self.values[index][:-1]
                case ActivationFunction.BinaryStep:
                    self.values[index][:-1] = np.where(\
                        self.values[index][:-1] >= 0, 1, 0)
                case ActivationFunction.Relu:
                    self.values[index][:-1] = np.where(\
                        self.values[index][:-1] >= 0, \
                        self.values[index][:-1], 0)
                case ActivationFunction.Sigmoid:
                    self.values[index][:-1] = \
                        1 / (1 + np.exp(-self.values[index][:-1]))
                case ActivationFunction.Softplus:
                    self.values[index][:-1] = \
                        np.log(1 + np.exp(self.values[index][:-1]))
                case other:
                    print("Not implemented: " \
                        f"{self.activation_functions[index]}")
        
        # Return output values.
        return self.values[-1]
    
    def generate_variation(self, mutation_chance, mutation_max):
        """
        Returns a new NeuralNetwork object in which every weight has had a 
        mutation_chance chance to be altered by at most mutation_max * 100 
        percent.
        Mutation chance must be a float between 0 and 1. Mutation max must be 
        a float, and is treated as a fraction of the original weight.
        For example a weight with a value of 0.5 will, with a mutation max of 
        0.1, be altered by at most 0.5 * 0.1 = 0.05.
        """
        # Check types.
        func_name = "NeuralNetwork.generate_variation()"
        
        # Check if mutation_chance is a float.
        assert isinstance(mutation_chance, float), f"{func_name}: " \
            "mutation_chance must be a float between 0 and 1."
        
        # Check if mutation_chance is between 0 and 1.
        assert mutation_chance >= 0 and mutation_chance <= 1, \
            f"{func_name}: mutation_chance must be a float between 0 and 1."
        
        # Check if mutation_max is a float.
        assert isinstance(mutation_max, float), f"{func_name}: " \
            "mutation_max must be a float."
        
        # Create new NeuralNetwork object.
        variation_nn = copy.deepcopy(self)
        
        # Loop over the weights.
        for weights in variation_nn.weights:
            if weights is None:
                continue
            
            for row_index, row in enumerate(weights[:-1]):
                for col_index, value in enumerate(row):
                    # Generate random number to decide whether to alter this 
                    # weight or not.
                    rand_num = np.random.rand(1)[0]
                    if rand_num < mutation_chance:
                        # Generate random number to decide the amount of 
                        # altering.
                        alter_factor = np.random.rand(1)[0] * 2 - 1
                        weights[row_index][col_index] = \
                            weights[row_index][col_index] * alter_factor \
                                * mutation_max
        
        # Return altered brain.
        return variation_nn

class ActivationFunction(Enum):
    
    Linear      = 0
    BinaryStep  = 1
    Relu        = 2
    Sigmoid     = 3
    Softplus    = 4
