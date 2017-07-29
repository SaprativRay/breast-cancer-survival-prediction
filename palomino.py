import numpy as np
import scipy as sp

default_settings = {
    "weights_low": -0.1,  # Lower bound on initial weight range
    "weights_high": 0.1,  # Upper bound on initial weight range
    "initial_bias_value": 0.01,
}

def softmax(signal, derivative=False):
    # Calculate activation signal
    e_x = np.exp(signal - np.max(signal, axis=1, keepdims=True))
    signal = e_x / np.sum(e_x, axis=1, keepdims=True)

    if derivative:
        return np.ones(signal.shape)
    else:
        # Return the activation signal
        return signal

def softmax_categorical_cross_entropy_cost(outputs, targets, derivative=False, epsilon=1e-11):
    """
    The output signals should be in the range [0, 1]
    """
    outputs = np.clip(outputs, epsilon, 1 - epsilon)

    if derivative:
        return outputs - targets
    else:
        return np.mean(-np.sum(targets * np.log(outputs), axis=1))

def add_bias(A):
    # Add a bias value of 1. The value of the bias is adjusted through
    # weights rather than modifying the input signal.
    return np.hstack(( np.ones((A.shape[0],1)), A ))
#end addBias

def learn(network, trainingset, testset, cost_function, ERROR_LIMIT=1e-3, max_iterations=(),
                              weight_step_max=50., weight_step_min=0., start_step=0.5, learn_max=1.2, learn_min=0.5,
                              print_rate=1000, save_trained_network=False):
    # Implemented according to iRprop+
    # http://sci2s.ugr.es/keel/pdf/algorithm/articulo/2003-Neuro-Igel-IRprop+.pdf

    training_data = np.array([instance.features for instance in trainingset])
    training_targets = np.array([instance.targets for instance in trainingset])
    test_data = np.array([instance.features for instance in testset])
    test_targets = np.array([instance.targets for instance in testset])

    # Storing the current / previous weight step size
    weight_step = [np.full(weight_layer.shape, start_step) for weight_layer in network.weights]

    # Storing the current / previous weight update
    dW = [np.ones(shape=weight_layer.shape) for weight_layer in network.weights]

    # Storing the previous derivative
    previous_dEdW = [1] * len(network.weights)

    # Storing the previous error measurement
    prev_error = ( )  # inf

    input_signals, derivatives = network.update(training_data, trace=True)
    out = input_signals[-1]
    cost_derivative = cost_function(out, training_targets, derivative=True).T
    delta = cost_derivative * derivatives[-1]
    error = cost_function(network.update(test_data), test_targets)

    n_samples = float(training_data.shape[0])
    layer_indexes = range(len(network.layers))[::-1]  # reversed
    epoch = 0

    while error > ERROR_LIMIT and epoch < max_iterations:
        epoch += 1

        for i in layer_indexes:
            # Loop over the weight layers in reversed order to calculate the deltas

            # Calculate the delta with respect to the weights
            dEdW = (np.dot(delta, add_bias(input_signals[i])) / n_samples).T

            if i != 0:
                """Do not calculate the delta unnecessarily."""
                # Skip the bias weight
                weight_delta = np.dot(network.weights[i][1:, :], delta)

                # Calculate the delta for the subsequent layer
                delta = weight_delta * derivatives[i - 1]

            # Calculate sign changes and note where they have changed
            diffs = np.multiply(dEdW, previous_dEdW[i])
            pos_indexes = np.where(diffs > 0)
            neg_indexes = np.where(diffs < 0)
            zero_indexes = np.where(diffs == 0)

            # positive
            if np.any(pos_indexes):
                # Calculate the weight step size
                weight_step[i][pos_indexes] = np.minimum(weight_step[i][pos_indexes] * learn_max, weight_step_max)

                # Calculate the weight step direction
                dW[i][pos_indexes] = np.multiply(-np.sign(dEdW[pos_indexes]), weight_step[i][pos_indexes])

                # Apply the weight deltas
                network.weights[i][pos_indexes] += dW[i][pos_indexes]

            # negative
            if np.any(neg_indexes):
                weight_step[i][neg_indexes] = np.maximum(weight_step[i][neg_indexes] * learn_min, weight_step_min)

                if error > prev_error:
                    # iRprop+ version of resilient backpropagation
                    network.weights[i][neg_indexes] -= dW[i][neg_indexes]  # backtrack

                dEdW[neg_indexes] = 0

            # zeros
            if np.any(zero_indexes):
                dW[i][zero_indexes] = np.multiply(-np.sign(dEdW[zero_indexes]), weight_step[i][zero_indexes])
                network.weights[i][zero_indexes] += dW[i][zero_indexes]

            # Store the previous weight step
            previous_dEdW[i] = dEdW
        # end weight adjustment loop

        prev_error = error

        input_signals, derivatives = network.update(training_data, trace=True)
        out = input_signals[-1]
        cost_derivative = cost_function(out, training_targets, derivative=True).T
        delta = cost_derivative * derivatives[-1]
        error = cost_function(network.update(test_data), test_targets)

        if epoch % print_rate == 0:
            # Show the current training status
            print            "[training] Current error:", error, "\tEpoch:", epoch

    print    "[training] Finished:"
    print    "[training]   Converged to error bound (%.4g) with error %.4g." % (ERROR_LIMIT, error)
    print    "[training]   Measured quality: %.4g" % network.measure_quality(training_data, training_targets, cost_function)
    print    "[training]   Trained for %d epochs." % epoch

class neuralnet:
    def __init__(self, settings):
        self.__dict__.update(default_settings)
        self.__dict__.update(settings)

        assert not softmax in map(lambda x: x[1], self.layers) or softmax == self.layers[-1][1], \
            "The `softmax` activation function may only be used in the final layer."

        # Count the required number of weights. This will speed up the random number generation when initializing weights
        self.n_weights = (self.n_inputs + 1) * self.layers[0][0] + \
                         sum((self.layers[i][0] + 1) * layer[0] for i, layer in enumerate(self.layers[1:]))

        # Initialize the network with new randomized weights
        self.set_weights(self.generate_weights(self.weights_low, self.weights_high))

        # Initalize the bias to 0.01
        for index in range(len(self.layers)):
            self.weights[index][:1, :] = self.initial_bias_value

    # end


    def generate_weights(self, low=-0.1, high=0.1):
        # Generate new random weights for all the connections in the network
        return np.random.uniform(low, high, size=(self.n_weights,))

    # end


    def set_weights(self, weight_list):
        # This is a helper method for setting the network weights to a previously defined list
        # as it's useful for loading a previously optimized neural network weight set.
        # The method creates a list of weight matrices. Each list entry correspond to the
        # connection between two layers.
        start, stop = 0, 0
        self.weights = []
        previous_shape = self.n_inputs + 1  # +1 because of the bias

        for n_neurons, activation_function in self.layers:
            stop += previous_shape * n_neurons
            self.weights.append(weight_list[start:stop].reshape(previous_shape, n_neurons))

            previous_shape = n_neurons + 1  # +1 because of the bias
            start = stop

    # end


    def get_weights(self, ):
        # This will stack all the weights in the network on a list, which may be saved to the disk.
        return [w for l in self.weights for w in l.flat]

    # end


    def error(self, weight_vector, training_data, training_targets, cost_function):
        # assign the weight_vector as the network topology
        self.set_weights(np.array(weight_vector))
        # perform a forward operation to calculate the output signal
        out = self.update(training_data)
        # evaluate the output signal with the cost function
        return cost_function(out, training_targets)

    # end


    def measure_quality(self, training_data, training_targets, cost_function):
        # perform a forward operation to calculate the output signal
        out = self.update(training_data)
        # calculate the mean error on the data classification
        mean_error = cost_function(out, training_targets) / float(training_data.shape[0])
        # calculate the numeric range between the minimum and maximum output value
        range_of_predicted_values = np.max(out) - np.min(out)
        # return the measured quality
        return 1 - (mean_error / range_of_predicted_values)

    # end


    def gradient(self, weight_vector, training_data, training_targets, cost_function):
        # assign the weight_vector as the network topology
        self.set_weights(np.array(weight_vector))

        input_signals, derivatives = self.update(training_data, trace=True)
        out = input_signals[-1]
        cost_derivative = cost_function(out, training_targets, derivative=True).T
        delta = cost_derivative * derivatives[-1]

        layer_indexes = range(len(self.layers))[::-1]  # reversed
        n_samples = float(training_data.shape[0])
        deltas_by_layer = []

        for i in layer_indexes:
            # Loop over the weight layers in reversed order to calculate the deltas
            deltas_by_layer.append(list((np.dot(delta, add_bias(input_signals[i])) / n_samples).T.flat))

            if i != 0:
                # i!= 0 because we don't want calculate the delta unnecessarily.
                weight_delta = np.dot(self.weights[i][1:, :], delta)  # Skip the bias weight

                # Calculate the delta for the subsequent layer
                delta = weight_delta * derivatives[i - 1]
        # end weight adjustment loop

        return np.hstack(reversed(deltas_by_layer))

    # end gradient

    def update(self, input_values, trace=False):
        # This is a forward operation in the network. This is how we
        # calculate the network output from a set of input signals.
        output = input_values

        if trace:
            derivatives = []  # collection of the derivatives of the act functions
            outputs = [output]  # passed through act. func.

        for i, weight_layer in enumerate(self.weights):
            # Loop over the network layers and calculate the output
            signal = np.dot(output, weight_layer[1:, :]) + weight_layer[0:1, :]  # implicit bias
            output = self.layers[i][1](signal)

            if trace:
                outputs.append(output)
                derivatives.append(
                    self.layers[i][1](signal, derivative=True).T)  # the derivative used for weight update

        if trace:
            return outputs, derivatives

        return output

    # end


    def predict(self, predict_set):
        predict_data = np.array([instance.features for instance in predict_set])
        return self.update(predict_data)
# end class

class Instance:
    def __init__(self, features, target=None):
        self.features = np.array(features)

        if target != None:
            self.targets = np.array(target)
        else:
            self.targets = None

def sigmoid(signal, derivative=False):
    # Prevent overflow.
    signal = np.clip(signal, -500, 500)

    # Calculate activation signal
    signal = sp.special.expit(signal)

    if derivative:
        # Return the partial derivation of the activation function
        return np.multiply(signal, 1 - signal)
    else:
        # Return the activation signal
        return signal
# end activation function

def cross_entropy_cost(outputs, targets, derivative=False, epsilon=1e-11):
    """
    The output signals should be in the range [0, 1]
    """
    # Prevent overflow
    outputs = np.clip(outputs, epsilon, 1 - epsilon)
    divisor = np.maximum(outputs * (1 - outputs), epsilon)

    if derivative:
        return (outputs - targets) / divisor
    else:
        return np.mean(-np.sum(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs), axis=1))


# end cost function

inputs = np.loadtxt('risks.csv', delimiter=',', comments='#', skiprows=1, dtype=None)
outputs = np.loadtxt('reports.csv', delimiter=',', comments='#', skiprows=1, dtype=None)

dataset = []
for i in xrange(0, len(outputs)):
    dataset += [Instance(inputs[i], outputs[i:i+1])]

settings       = {
    "n_inputs" : 16,
    "layers"   : [  (16, sigmoid), (8, sigmoid), (1, sigmoid) ]
}

network        = neuralnet( settings )
training_set   = dataset
test_set       = dataset
cost_function  = cross_entropy_cost
from nimblenet.learning_algorithms import *
learn(network, training_set, test_set, cost_function,
                ERROR_LIMIT=0.09,
                #max_iterations=1000,
                )