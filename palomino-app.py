import numpy as np
import cPickle

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

    def generate_weights(self, low=-0.1, high=0.1):
        # Generate new random weights for all the connections in the network
        return np.random.uniform(low, high, size=(self.n_weights,))

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

    def predict(self, predict_set):
        predict_data = np.array([instance.features for instance in predict_set])
        return self.update(predict_data)

class Instance:
    def __init__(self, features, target=None):
        self.features = np.array(features)

        if target != None:
            self.targets = np.array(target)
        else:
            self.targets = None

network = neuralnet({"n_inputs": 1, "layers": [[0, None]]})

with open('network.dat', 'rb') as file:
    store_dict = cPickle.load(file)

network.n_inputs = store_dict["n_inputs"]
network.n_weights = store_dict["n_weights"]
network.layers = store_dict["layers"]
network.weights = store_dict["weights"]

# TODO more user-friendly input-output mechanisms

a1 = input("Diagnosis: ")
a2 = input("Forced vital capacity (FVC): ")
a3 = input("Volume exhaled at the end of first second of forced expiration: ")
a4 = input("Zubrod Score: ")
a5 = input("Pain (Y: 1, N: 0): ")
a6 = input("Haemoptysis (1/0): ")
a7 = input("Dyspnoea (1/0): ")
a8 = input("Cough (1/0): ")
a9 = input("Weakness (1/0): ")
a10 = input("T in clinical TNM, size of original tumour (11-14): ")
a11 = input("Type 2 Diabetes Mellitus (1/0): ")
a12 = input("MI up to 6 months (1/0): ")
a13 = input("Peripheral arterial diseases (1/0): ")
a14 = input("Smoking (1/0): ")
a15 = input("Asthma (1/0): ")
a16 = input("Age: ")

prediction_set = [Instance([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16])]
print("Predicted Mortality: %d%" % network.predict(prediction_set) * 100)
