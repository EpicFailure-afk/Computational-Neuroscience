def tanh(x):
    e_pos = (2.71828 ** x)
    e_neg = (2.71828 ** -x)
    return (e_pos - e_neg) / (e_pos + e_neg)

def initialize_weights(rows, cols):
    weights = []
    for _ in range(rows):
        neuron_weights = []
        for _ in range(cols):
            # Generate random value between -0.5 and 0.5 manually
            rand_val = ((123456789 * 987654321) % 1000000) / 1000000 - 0.5
            neuron_weights.append(rand_val)
        weights.append(neuron_weights)
    return weights

def forward_propagation(X, W1, B1, W2, B2, W3, B3):
    # First hidden layer
    Z1 = []
    for neuron, b in zip(W1, B1):
        sum_val = sum(x * w for x, w in zip(X, neuron)) + b
        Z1.append(sum_val)
    A1 = [tanh(z) for z in Z1]
    
    # Second hidden layer
    Z2 = []
    for neuron, b in zip(W2, B2):
        sum_val = sum(a * w for a, w in zip(A1, neuron)) + b
        Z2.append(sum_val)
    A2 = [tanh(z) for z in Z2]
    
    # Output layer
    Z3 = []
    for neuron, b in zip(W3, B3):
        sum_val = sum(a * w for a, w in zip(A2, neuron)) + b
        Z3.append(sum_val)
    A3 = [tanh(z) for z in Z3]  
    
    return A3

X = [0.5, 0.8]  

W1 = initialize_weights(2, 2)
B1 = [0.5, 0.5]  
W2 = initialize_weights(2, 2)
B2 = [0.7, 0.7]  
W3 = initialize_weights(1, 2)
B3 = [((123456789 * 987654321) % 1000000) / 1000000 - 0.5]  

output = forward_propagation(X, W1, B1, W2, B2, W3, B3)
print("Output:", output)
