def custom_random(low, high, seed):
    seed = (seed * 9301 + 49297) % 233280
    return low + (seed / 233280.0) * (high - low)

def custom_exponential(x, terms):
    res = 1
    fact = 1
    pow_x = 1
    count = 1
    while count < terms:
        fact *= count
        pow_x *= x
        res += pow_x / fact
        count += 1
    return res

def custom_tanh(x):
    exp_x = custom_exponential(x, 10)
    exp_neg_x = custom_exponential(-x, 10)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

seed = 42
values = []
inc = 0
while inc < 6:
    values.append(custom_random(-0.5, 0.5, seed + inc))
    inc += 1

w1, w2, w3, w4, w5, w6 = values
b1, b2 = 0.5, 0.7
i1, i2 = 0.05, 0.1
target_o1, target_o2 = 0.1, 0.99

h1_input = w1 * i1 + w2 * i2 + b1
h2_input = w3 * i1 + w4 * i2 + b1
h1_output = custom_tanh(h1_input)
h2_output = custom_tanh(h2_input)

o1_input = w5 * h1_output + w6 * h2_output + b2
o2_input = w5 * h1_output + w6 * h2_output + b2
o1_output = custom_tanh(o1_input)
o2_output = custom_tanh(o2_input)

E_o1 = 0.5 * (target_o1 - o1_output) * (target_o1 - o1_output)
E_o2 = 0.5 * (target_o2 - o2_output) * (target_o2 - o2_output)
total_loss = E_o1 + E_o2

print("Hidden Outputs: h1 =", h1_output, ", h2 =", h2_output)
print("Output Values: o1 =", o1_output, ", o2 =", o2_output)
print("Losses: E_o1 =", E_o1, ", E_o2 =", E_o2)
print("Total Loss:", total_loss)


