import sys
import math
import numpy as np
import neural_network as nn
import trainer as t
import data as d

def main (args=None):
    if args is None:
        args = sys.argv[1:]

    # gather data set
    dataX = d.x
    dataY = d.y

    # shuffle to ensure random testing data set
    np.random.shuffle(dataX)
    np.random.shuffle(dataY)

    # split up the data set
    training_split = .5
    train_x_index = int(math.floor(len(dataX)*training_split))
    train_y_index = int(math.floor(len(dataY)*training_split))
    test_x_index = int(math.floor(len(dataX)*(1-training_split)))
    test_y_index = int(math.floor(len(dataY)*(1-training_split)))

    # training data
    trainX = np.array(tuple(dataX[:train_x_index]), dtype=float)
    trainY = np.array(tuple(dataY[:train_y_index]), dtype=float)

    # testing data
    testX = np.array(tuple(dataX[test_x_index:]), dtype=float)
    testY = np.array(tuple(dataY[train_y_index:]), dtype=float)

    # normalize inputs
    trainX = trainX/np.amax(trainX, axis=0)
    testX = testX/np.amax(testX, axis=0)

    # evaluate number of input/ouput neurons
    inputs = len(trainX[0])
    outputs = len(trainY[0])

    # train neural net
    NN = nn.Neural_Network(inputs, outputs, Lambda=0.001)
    T = t.trainer(NN)
    T.train(trainX, trainY, testX, testY)

    # print results
    results = NN.forward(testX)
    i = 0
    for r in results:
        line = "Prediction: " + str("{:4.2f}").format(r[0]) + "; "
        line += "Actual: " + str("{:4.2f}").format(testY[i][0]) + "; "
        line += "Delta: " + str("{:4.2f}").format(testY[i][0] - r[0]) + "; "
        print line
        i = i + 1

if __name__ == "__main__":
    main()
