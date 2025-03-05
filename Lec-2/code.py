def generate_random(low, high, seed):
    seed = (seed * 9301 + 49297) % 233280
    return low + (seed / 233280) * (high - low)


def calculate_exponential(x, terms=10):
    result = 1
    factorial = 1
    power = 1
    i = 1
    while i < terms:
        factorial *= i
        power *= x
        result += power / factorial
        i += 1
    return result


def tanh_activation(x):
    exp_x = calculate_exponential(x)
    exp_neg_x = calculate_exponential(-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)


# Initialize weights and biases
seed = 42
weights = [generate_random(-0.5, 0.5, seed + i) for i in range(6)]
b1 = 0.5
b2 = 0.7
inputs = [0.05, 0.1]
targets = [0.1, 0.99]

w1, w2, w3, w4, w5, w6 = weights
i1, i2 = inputs
target_o1, target_o2 = targets

# Forward Pass
print("+-------------------------------+")
print("|        Forward Pass           |")
print("+-------------------------------+")

h1 = tanh_activation(w1 * i1 + w2 * i2 + b1)
h2 = tanh_activation(w3 * i1 + w4 * i2 + b1)

o1 = tanh_activation(w5 * h1 + w6 * h2 + b2)
o2 = tanh_activation(w5 * h1 + w6 * h2 + b2)

E_o1 = 0.5 * (target_o1 - o1) * (target_o1 - o1)
E_o2 = 0.5 * (target_o2 - o2) * (target_o2 - o2)
total_loss = E_o1 + E_o2

print("| Hidden Outputs | h1 =", h1, "| h2 =", h2, "|")
print("| Output Values  | o1 =", o1, "| o2 =", o2, "|")
print("| Losses         | E_o1 =", E_o1, "| E_o2 =", E_o2, "|")
print("| Total Loss     |", total_loss, "|")
print("+-------------------------------+")

# Backward Pass
print("\n+-------------------------------+")
print("|        Backward Pass          |")
print("+-------------------------------+")

delta_o1 = -(target_o1 - o1) * (1 - o1 * o1)
delta_o2 = -(target_o2 - o2) * (1 - o2 * o2)

delta_h1 = delta_o1 * w5 * (1 - h1 * h1) + delta_o2 * w5 * (1 - h1 * h1)
delta_h2 = delta_o1 * w6 * (1 - h2 * h2) + delta_o2 * w6 * (1 - h2 * h2)

grad_w1 = delta_h1 * i1 * (1 - h1 * h1)
grad_w2 = delta_h1 * i2 * (1 - h1 * h1)
grad_w3 = delta_h2 * i1 * (1 - h2 * h2)
grad_w4 = delta_h2 * i2 * (1 - h2 * h2)
grad_w5 = delta_o1 * h1 * (1 - o1 * o1)
grad_w6 = delta_o2 * h2 * (1 - o2 * o2)

grad_b1 = delta_h1 * (1 - h1 * h1) + delta_h2 * (1 - h2 * h2)
grad_b2 = delta_o1 + delta_o2

learning_rate = 0.1

w1 -= learning_rate * grad_w1
w2 -= learning_rate * grad_w2
w3 -= learning_rate * grad_w3
w4 -= learning_rate * grad_w4
w5 -= learning_rate * grad_w5
w6 -= learning_rate * grad_w6
b1 -= learning_rate * grad_b1
b2 -= learning_rate * grad_b2

print("+-------------------------------+")
print("|  Updated Weights and Biases   |")
print("+-------------------------------+")
print("| w1 =", w1, "| w2 =", w2, "|")
print("| w3 =", w3, "| w4 =", w4, "|")
print("| w5 =", w5, "| w6 =", w6, "|")
print("| b1 =", b1, "| b2 =", b2, "|")
print("+-------------------------------+")
