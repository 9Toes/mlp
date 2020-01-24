'''
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
'''
import numpy as np
import math as m
import matplotlib.pyplot as plt
import sys

class mlp:
    def __init__(self, inputs, targets, nhidden):
        self.beta = 1
        self.eta = 0.3
        self.momentum = 0.9

        self.inputs = inputs
        self.targets = targets
        self.nhidden = nhidden
        self.noutput = 8

        self.low = 0
        self.high = 0

        self.init_weights(inputs)


    def init_weights(self, inputs):
        # −1/√n < w < 1/√n, where w is weight, and n is the number of nodes in the input layer to those weights
        # 2D numpy array weights hidden [j_hidden][i_input]
        # 2D numpy array weights output [k_output][j_hidden]

        self.low = -1/np.sqrt(len(inputs[0]))
        self.high = 1/np.sqrt(len(inputs[0]))
        
        #inputs already initialised with bias
        self.weights_hidden = np.random.uniform(self.low, self.high, (self.nhidden, len(inputs[0])))
        self.weights_output = np.random.uniform(self.low, self.high, (self.noutput, self.nhidden + 1))


    def earlystopping(self, inputs, targets, valid, validtargets, test, test_targets):
        inputs = self.add_bias(inputs)
        valid = self.add_bias(valid)
        error_train = []
        error_valid = []

        percentages = []

        cross_valid_inputs = np.concatenate((inputs, valid))
        cross_valid_targets = np.concatenate((targets, validtargets))

        k = 12
        
        inputs_len = int(len(cross_valid_inputs)/k)
        targets_len = int(len(cross_valid_inputs)/k)

        best_val_error = sys.maxsize
        best_hidden = self.weights_hidden
        best_output = self.weights_output

        iter = 0
        while iter < k:
            valid = cross_valid_inputs[:inputs_len]
            validtargets = cross_valid_targets[:targets_len]

            train = cross_valid_inputs[inputs_len:]
            traintargets = cross_valid_targets[targets_len:]

            self.weights_hidden = 0
            self.weights_output = 0

            self.init_weights(train)

            #Code from book
            old_val_error1 = 100002 
            old_val_error2 = 100001 
            new_val_error = 100000
            i = 0

            while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1) > 0.001)):
                trainout_a = self.train(inputs, targets)
                train_error = self.calculate_error_output(trainout_a, targets, inputs)
                error_train.append(train_error)
                old_val_error2 = old_val_error1
                old_val_error1 = new_val_error
                validout_a, validout_a = self.forward(valid)
                new_val_error = self.calculate_error_output(validout_a, validtargets, valid)
                error_valid.append(new_val_error)
                i += 1

            print("\nStopped. {0}, {1}, {2}".format(new_val_error, old_val_error1, old_val_error2))

            correct = self.confusion(test, test_targets)
            percentages.append(correct)

            if new_val_error < best_val_error:
                best_val_error = new_val_error
                best_hidden = self.weights_hidden
                best_output = self.weights_output

            print("iteration {0}, best_val_error {1}".format(iter, best_val_error))  

            cross_valid_inputs = np.concatenate((train, valid))
            cross_valid_targets = np.concatenate((traintargets, validtargets))

            iter += 1
        
        self.weights_hidden = best_hidden
        self.weights_output = best_output

        print(percentages)
        avg_percent = np.sum(percentages)/len(percentages)
        print("Average correct classes: {0} percent.".format(avg_percent))
        stdev = np.std(percentages)
        print("Standard deviation: {0} percent.".format(stdev))

        # plt.plot(error_train, label='train')
        # plt.plot(error_valid, label='valid')
        # plt.legend()
        # plt.show()

        # self.train(inputs, targets)


    def calculate_error_output(self, output_a, targets, inputs):
        return np.sum(((output_a-targets)**2)/2)/len(inputs)


    def calculate_delta_hidden(self, hidden_a, delta_o):
        return hidden_a*self.beta*(1.0-hidden_a)*(np.dot(delta_o, self.weights_output))


    def calculate_delta_output(self, output_a, targets):
        return output_a-targets


    def train(self, inputs, targets, iterations=100):
        update_weights_h = np.zeros((np.shape(np.transpose(self.weights_hidden))))
        update_weights_o = np.zeros((np.shape(np.transpose(self.weights_output))))

        for iter in range(iterations):
            hidden_a, output_a = self.forward(inputs)  

            delta_o = self.calculate_delta_output(output_a, targets)
            delta_h = self.calculate_delta_hidden(hidden_a, delta_o)

            update_weights_h = (self.eta*(np.dot(np.transpose(inputs), delta_h[:,1:])) + (self.momentum*update_weights_h))/len(inputs)
            update_weights_o = (self.eta*(np.dot(np.transpose(hidden_a), delta_o)) + (self.momentum*update_weights_o))/len(inputs)

            self.weights_hidden -= np.transpose(update_weights_h)
            self.weights_output -= np.transpose(update_weights_o)

        return output_a


    def add_bias(self, matrix):
        return np.concatenate((-np.ones((len(matrix), 1)), matrix), axis=1)


    def sigmoid(self, matrix):
        return 1.0/(1.0 + np.exp(-self.beta * matrix))


    def forward(self, inputs):
        hidden_h = np.dot(inputs, np.transpose(self.weights_hidden))
        hidden_a = self.sigmoid(hidden_h)
        hidden_a = self.add_bias(hidden_a)
        output_h = np.dot(hidden_a, np.transpose(self.weights_output))
        output_a = np.copy(output_h) #Linear activation function

        return hidden_a, output_a


    def confusion(self, inputs, targets):
        inputs = self.add_bias(inputs)
        
        hidden_a, output_a = self.forward(inputs)

        matrix = np.zeros((len(targets[0]), len(output_a[0])))

        for x, y in zip(targets, output_a):
            matrix[x.argmax(), y.argmax()] += 1
        
        print(matrix)

        correct = (np.trace(matrix) / np.sum(matrix)) * 100
        print("{0} percent correct classes.".format(correct))
        return correct
         
