# Container types:
# 1D vector
# 2D matrix
# 3D matrix
# tensor - object that can be represented as an array, but isn't just an array
# for coding, tensors will be used and represented as arrays

# dot product
import numpy as np
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = np.dot(weights, inputs) + bias
output1 = inputs[0]*weights[0]+inputs[1]*weights[1]+inputs[2]*weights[2]+inputs[3]*weights[3] + bias

print(output)
print(output1)

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -.91, .26, -.5],
           [-.26, -.27, .17, 0.87]]
# sort of like the m in y = mx+b
biases = [2, 3, 0.5]
# sort of like the b in y = mx+b

# conduct the dot product between weights and inputs matrices. Then add bias
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output = n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)
output = np.dot(weights, inputs) + biases
print(output)
