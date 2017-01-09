import numpy as np

class Neural_Network(object):
    def __init__(self, inputs, outputs, Lambda=0):
        # Define Hyperparameters
        self.inputLayerSize = inputs
        self.outputLayerSize = outputs
        self.hiddenLayerSize = 3

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

        # Regularization Parameter:
        self.Lambda = Lambda

    # hidden layer activation function
    def afhl(self, x):
        # sigmoid function
        return 1/(1+np.exp(-x))

    # hidden layer activation function prime
    def afhlPrime(self, x):
        # gradient of sigmoid
        return np.exp(-x)/((1+np.exp(-x))**2)

    # ouput layer activation function
    def afol(self, x):
        # linear function
        return x
        # sigmoid function
        #return 1/(1+np.exp(-x))

    # Propogate inputs though network
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.afhl(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.afol(self.z3)
        return yHat

    # Compute cost for given X,y, use weights already stored in class.
    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5*sum((y - self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J

    # Compute derivative with respect to W and W2 for a given X and y:
    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.afol(self.z3))
        # Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2

        delta2 = np.dot(delta3, self.W2.T)*self.afhlPrime(self.z2)
        # Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1

        return dJdW1, dJdW2

    # Helper functions for interacting with other methods/classes
    # Get W1 and W2 Rolled into vector:
    def getParams(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    # Set W1 and W2 using single parameter vector:
    def setParams(self, params):
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
