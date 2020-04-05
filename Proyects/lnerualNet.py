#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:06:44 2020

@author: josej
"""

import numpy as np
import matplotlib.pyplot as plt


class lNerualNet:
    
    
    
    activatorFunctions = {"sigmoid": cls.sigmoid}
    
    
    def __init__(self, randomSeed=0, multIndex=.01):
        self._randomSeed = randomSeed
        self._multIndex = multIndex
        
    @property
    def randomSeed(self):
        return self._randomSeed
    
    
    @randomSeed.setter
    def randomSeed(self, value):
        if value < 0 or value > 2**32 - 1:
            raise ValueError("Seed must be between 0 and 2**32 - 1")
        self._randomSeed = value
        
        
    def activate_randomSeed(self):
        """
        Sets the np.random.seed with the current randomSeed value

        Returns
        -------
        None.
        """
        
        np.random.seed(self.randomSeed)
    
    
    @property
    def multIndex(self):
        return self._multIndex
    
    
    @multIndex.setter
    def multIndex(self, value):
        self._multIndex = value
    
    
    @classmethod
    def sigmoid(self, Z):
        """
        Implements the sigmoid activation in numpy
        
        Arguments:
        Z -- numpy array of any shape
        
        Returns:
        A -- output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
        """
        
        A = 1/(1+np.exp(-Z))
        cache = Z
        
        return A, cache
    
    
    @classmethod
    def relu(self, Z):
        """
        Implement the RELU function.
        Arguments:
        Z -- Output of the linear layer, of any shape
        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
        """
        
        A = np.maximum(0,Z)
        
        assert(A.shape == Z.shape)
        
        cache = Z 
        return A, cache
    
    
    @classmethod
    def relu_backward(self, dA, Z):
        """
        Implement the backward propagation for a single RELU unit.
        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently
        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        
        assert (dZ.shape == Z.shape)
        
        return dZ
    
    
    @classmethod
    def sigmoid_backward(self, dA, Z):
        """
        Implement the backward propagation for a single SIGMOID unit.
        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently
        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        assert (dZ.shape == Z.shape)
        
        return dZ
    
    
    def initialize_parameters(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        
        parameters = {}
        
        for l in range(1, len(layer_dims)):
            parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * self.multIndex
            parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
            
        return parameters
    
    
    @staticmethod
    def linear_forward(A, W, b):
        """
        Implement the linear part of a layer's forward propagation.
    
        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
    
        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        
        Z = np.dot(W, A)  + b
        cache = (A, W, b)
        
        return Z, cache
    
    
    
    def train(self, X, Y, layers_dims=[X.shape[0],1], learning_rate = 0.001, num_iterations = 3000, print_cost=False, autoTune=True):
        
        if not autoTune:
            return train_inner(X, Y, layers_dims, learning_rate, num_iterations, print_cost)
    
    
    def train_inner(self, X, Y, layers_dims, learning_rate, num_iterations, print_cost):
        pass
        