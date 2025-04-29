Sentence = ["I", "use", "Arch", "btw"]
Sen_size = len(Sentence)

# word-to-index mapping
word_to_idx = {}
for i in range(Sen_size):
    word_to_idx[Sentence[i]] = i

# one-hot encode
def one_hot_encode(word_idx, Sen_size):
    encoding = [0] * Sen_size
    encoding[word_idx] = 1
    return encoding

# training data
X = [
    [word_to_idx["I"], word_to_idx["use"], word_to_idx["Arch"]]
]
Y = [word_to_idx["btw"]]

# Matrix multiplication
def matrix_multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Incompatible matrix dimensions")
    
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

# Vector-matrix multiplication (for 1D vector)
def vector_matrix_multiply(v, A):
    result = [0] * len(A[0])
    for i in range(len(A[0])):
        for j in range(len(v)):
            result[i] += v[j] * A[j][i]
    return result

# Element-wise addition
def add_vectors(a, b):
    return [a[i] + b[i] for i in range(len(a))]

# Tanh activation function
def tanh(x):
    # Implement tanh without using built-in functions
    if x > 20:  # To avoid overflow
        return 1.0
    if x < -20:  # To avoid underflow
        return -1.0
    
    exp_x = 1.0
    exp_neg_x = 1.0
    
    term = 1.0
    for i in range(1, 20):  
        term *= x / i
        exp_x += term
        if term < 1e-10:  
            break
    
    term = 1.0
    for i in range(1, 20):
        term *= -x / i
        exp_neg_x += term
        if term < 1e-10:  
            break
    
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

# Softmax function
def softmax(x):
    max_val = x[0]
    for val in x:
        if val > max_val:
            max_val = val
    
    exp_values = []
    sum_exp = 0
    for val in x:
        exp_val = 1.0
        power = val - max_val
        term = 1.0
        for i in range(1, 20):  
            term *= power / i
            exp_val += term
            if term < 1e-10:  
                break
        exp_values.append(exp_val)
        sum_exp += exp_val
    
    # Normalize
    softmax_values = [exp_val / sum_exp for exp_val in exp_values]
    return softmax_values

# RNN parameters
hidden_size = 4

# Initialize weights with better initialization
def initialize_weights():
    # Use smaller initial weights to prevent exploding gradients
    # Wx: input to hidden
    Wx = [[0.005 * ((i * j) % 10 - 5) for j in range(hidden_size)] for i in range(Sen_size)]
    
    # Wh: hidden to hidden
    Wh = [[0.005 * ((i + j) % 10 - 5) for j in range(hidden_size)] for i in range(hidden_size)]
    
    # Wy: hidden to output
    Wy = [[0.005 * ((i - j) % 10 - 5) for j in range(Sen_size)] for i in range(hidden_size)]
    
    # Biases
    bh = [0.0 for i in range(hidden_size)]  
    by = [0.0 for i in range(Sen_size)]   
    
    return Wx, Wh, Wy, bh, by

# Forward propagation 
def forward_propagation(inputs, h_prev, Wx, Wh, Wy, bh, by):
    x_values = []
    h_values = [h_prev]
    y_values = []
    
    for t in range(len(inputs)):
        x_t = one_hot_encode(inputs[t], Sen_size)
        x_values.append(x_t)
        
        # Calculate hidden state
        wx_xt = vector_matrix_multiply(x_t, Wx)
        wh_hprev = vector_matrix_multiply(h_values[t], Wh)
        
        a_t = add_vectors(add_vectors(wx_xt, wh_hprev), bh)
        h_t = [tanh(val) for val in a_t]
        h_values.append(h_t)
        
        # Calculate output
        wy_ht = vector_matrix_multiply(h_t, Wy)
        y_t = softmax(add_vectors(wy_ht, by))
        y_values.append(y_t)
    
    return x_values, h_values, y_values

# Calculate loss 
def calculate_loss(y_pred, y_true):
    epsilon = 1e-15  # To avoid log(0)
    y_pred_clipped = [max(min(p, 1.0 - epsilon), epsilon) for p in y_pred]
    return -1 * (0.0 + y_pred_clipped[y_true])

# Backpropagation (BPTT) 
def backpropagation(x_values, h_values, y_values, targets, Wx, Wh, Wy):
    dWx = [[0 for _ in range(hidden_size)] for _ in range(Sen_size)]
    dWh = [[0 for _ in range(hidden_size)] for _ in range(hidden_size)]
    dWy = [[0 for _ in range(Sen_size)] for _ in range(hidden_size)]
    dbh = [0 for _ in range(hidden_size)]
    dby = [0 for _ in range(Sen_size)]
    
    # Initialize delta for the output layer
    # For softmax with cross-entropy loss
    dy = y_values[-1].copy()  
    dy[targets] -= 1.0  
    
    # Initialize delta for the hidden layer
    dh_next = [0] * hidden_size
    
    # Backpropagate through time
    T = len(x_values)
    for t in reversed(range(T)):
        # Gradient for output weights
        for i in range(hidden_size):
            for j in range(Sen_size):
                dWy[i][j] += h_values[t+1][i] * dy[j]
        
        # Gradient for output bias
        dby = add_vectors(dby, dy)
        
        # Gradient for hidden layer
        dh = vector_matrix_multiply(dy, [list(row) for row in zip(*Wy)])  # Transpose Wy
        dh = add_vectors(dh, dh_next)
        
        # Gradient for tanh
        dtanh = [0] * hidden_size
        for i in range(hidden_size):
            dtanh[i] = (1 - h_values[t+1][i]**2) * dh[i]
        
        # Gradient for hidden bias
        dbh = add_vectors(dbh, dtanh)
        
        # Gradient for hidden weights
        for i in range(hidden_size):
            for j in range(hidden_size):
                dWh[i][j] += h_values[t][i] * dtanh[j]
        
        # Gradient for input weights
        for i in range(Sen_size):
            for j in range(hidden_size):
                dWx[i][j] += x_values[t][i] * dtanh[j]
        
        # Gradient for previous hidden state
        dh_next = vector_matrix_multiply(dtanh, [list(row) for row in zip(*Wh)])  
    
    return dWx, dWh, dWy, dbh, dby

# Update weights 
def update_weights(Wx, Wh, Wy, bh, by, dWx, dWh, dWy, dbh, dby, learning_rate):
    # Gradient clipping
    def clip_gradient(grad, threshold=5.0):
        # Find the maximum absolute value in the gradient
        max_abs = 0.0
        for row in grad:
            for val in row:
                if abs(val) > max_abs:
                    max_abs = abs(val)
        
        # If the maximum is above threshold, scale down
        if max_abs > threshold:
            scale = threshold / max_abs
            return [[val * scale for val in row] for row in grad]
        return grad
    
    # Clip gradients
    dWx = clip_gradient(dWx)
    dWh = clip_gradient(dWh)
    dWy = clip_gradient(dWy)
    
    # Update weights using gradient descent
    for i in range(len(Wx)):
        for j in range(len(Wx[0])):
            Wx[i][j] -= learning_rate * dWx[i][j]
    
    for i in range(len(Wh)):
        for j in range(len(Wh[0])):
            Wh[i][j] -= learning_rate * dWh[i][j]
    
    for i in range(len(Wy)):
        for j in range(len(Wy[0])):
            Wy[i][j] -= learning_rate * dWy[i][j]
    
    for i in range(len(bh)):
        bh[i] -= learning_rate * dbh[i]
    
    for i in range(len(by)):
        by[i] -= learning_rate * dby[i]
    
    return Wx, Wh, Wy, bh, by

# Training loop with improved parameters
def train_rnn(X, Y, epochs=1000, learning_rate=0.01):  # Increased epochs, reduced learning rate
    Wx, Wh, Wy, bh, by = initialize_weights()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for i in range(len(X)):
            # Initialize hidden state
            h_prev = [0] * hidden_size
            
            # Forward pass 
            x_values, h_values, y_values = forward_propagation(X[i], h_prev, Wx, Wh, Wy, bh, by)
            
            # Calculate loss
            loss = calculate_loss(y_values[-1], Y[i])
            total_loss += loss
            
            # Backward pass 
            dWx, dWh, dWy, dbh, dby = backpropagation(x_values, h_values, y_values, Y[i], Wx, Wh, Wy)
            
            # Update weights
            Wx, Wh, Wy, bh, by = update_weights(Wx, Wh, Wy, bh, by, dWx, dWh, dWy, dbh, dby, learning_rate)
        
        # Implement learning rate decay
        if epoch > 0 and epoch % 200 == 0:
            learning_rate *= 0.5
            print(f"Reducing learning rate to {learning_rate}")
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")
            
            # Check prediction during training
            if epoch % 200 == 0:
                test_input = [word_to_idx["I"], word_to_idx["use"], word_to_idx["Arch"]]
                h_prev = [0] * hidden_size
                
                _, _, y_values = forward_propagation(test_input, h_prev, Wx, Wh, Wy, bh, by)
                probs = y_values[-1]
                print(f"Current probabilities: {Sentence[0]}: {probs[0]:.4f}, {Sentence[1]}: {probs[1]:.4f}, {Sentence[2]}: {probs[2]:.4f}, {Sentence[3]}: {probs[3]:.4f}")
    
    return Wx, Wh, Wy, bh, by

# Predict function 
def predict(inputs, Wx, Wh, Wy, bh, by):
    h_prev = [0] * hidden_size
    _, _, y_values = forward_propagation(inputs, h_prev, Wx, Wh, Wy, bh, by)
    
    # Get the index of the word with highest probability
    predicted_idx = y_values[-1].index(max(y_values[-1]))
    return predicted_idx, y_values[-1]

# Train the RNN
print("Training RNN...")
Wx, Wh, Wy, bh, by = train_rnn(X, Y, epochs=1000, learning_rate=0.01)

# Test the RNN
test_input = [word_to_idx["I"], word_to_idx["use"], word_to_idx["Arch"]]
predicted_idx, probabilities = predict(test_input, Wx, Wh, Wy, bh, by)
predicted_word = Sentence[predicted_idx]

print(f"\nInput: 'I use Arch'")
print(f"====> Predicted 4th word: '--> '{predicted_word}' <--'")
print(f"Expected 4th word: 'btw'")
print("\nProbabilities for each word:")
for i, word in enumerate(Sentence):
    print(f"{word}: {probabilities[i]:.4f}")