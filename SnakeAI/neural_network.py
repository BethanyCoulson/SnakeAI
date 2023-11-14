# NEURAL NETWORK
import random, math

def dot(vec1, vec2): # Returns the dot product of two vectors 
    dot_prod = 0
    for i in range(len(vec1)):
        dot_prod += vec1[i]*vec2[i]
    return dot_prod

def sigmoid(x):
    return 1/(1 + math.exp(-x))   

class NeuralNetwork:
    def __init__(self, shape, data):
        self.shape = shape
        self.data = data

    def propagate_forward(self, inp):
        vectors = [inp] # Creates a list to keep track of the vectors created at each layer
        for i in range(len(self.shape)-1): # Loops over each layer
            newVector = [] 
            for j in range(self.shape[i+1]): # Loops for each node calculation in the layer
                currentNode = self.data[i][j]
                newVector.append(sigmoid(dot(vectors[-1],currentNode))) # Calculates the value for the node and adds it to the vector for that layer
            vectors.append(newVector)
        return newVector

    def flatten(self):
        '''Iterates over each node in the neural network and appends it to a new list. This makes it easier to manipulate the network.''' 
        flattened = []
        for layer in self.data: 
            for node in layer:
                for edge in node:
                    flattened.append(edge)
        return flattened

    @classmethod
    def unflatten(cls, shape, data):
        unflattened_data = []
        for i in range(len(shape)-1):
            layer = []
            for j in range(shape[i+1]):
                edges = []
                for k in range(shape[i]):
                    edges.append(data.pop(0))
                layer.append(edges)
            unflattened_data.append(layer)
        return NeuralNetwork(shape, unflattened_data)

    @classmethod
    def random(cls, shape):
        '''Creates a new neural network with a specified shape and random weights for each edge.'''
        data = []
        for i in range(len(shape)-1): # Loops through all layers after the input layer
            layer = []
            for j in range(shape[i+1]): # Loops through all nodes in the layer
                node = []
                for k in range(shape[i]): # Loops through all the inputs for the node
                    node.append(2*random.random()-1) # Generates a random number between -1 and 1 for the weight of the edge
                layer.append(node)
            data.append(layer)
        return NeuralNetwork(shape, data) # Creates a NeuralNetwork object with the inputted shape and the random data
