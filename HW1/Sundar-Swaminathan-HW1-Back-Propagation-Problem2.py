#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1-x)

x0 = 1.0
x1, x2, x3, x4 = 1.0, 1.0, 0.0, 1.0
t5, t6, t7 = 0.2, 0.1, -0.3
y = 1.0

w15, w16 = 0.3, 0.1
w25, w26 = -0.2, 0.4
w35, w36 = 0.2, -0.3
w45, w46 = 0.1, 0.4
w57, w67 = -0.3, 0.2

learning_rate = 0.8

error = float('inf')
epoch = 0
while epoch <= 6:
    print("---new epoch---")
    # Calculate Hidden outputs
    y5 = w15 * x1 + w25 * x2 + w35 * x3 + w45 * x4 + x0 * t5
    print(f"y5 = {y5}")
    y6 = w16 * x1 + w26 * x2 + w36 * x3 + w46 * x4 + x0 * t6
    print(f"y6 = {y6}")
    h5 = sigmoid(y5)
    print(f"h5 = {h5}")
    h6 = sigmoid(y6)
    print(f"h6 = {h6}")

    # Calculate Network output
    z7 = h5 * w57 + h6 * w67 + x0 * t7
    print(f"z7 = {z7}")
    o7 = sigmoid(z7)
    error = y - o7
    print(f"o7 = {o7}, error = {error}")

    # Calcualte Error
    delta_o7 = error * sigmoid_derivative(o7)

    # Backpropagate Error
    error_h5 = delta_o7 * w57
    error_h6 = delta_o7 * w67
    delta_h5 = error_h5 * sigmoid_derivative(h5)
    delta_h6 = error_h6 * sigmoid_derivative(h6)

    # New weights
    w57 = w57 + (learning_rate * delta_o7 * h5)
    print(f"w57(new)={w57}")
    w67 = w67 + (learning_rate * delta_o7 * h5)
    print(f"w67(new)={w67}")
    w46 = w46 + (learning_rate * delta_h6 * h6)
    print(f"w46(new)={w46}")
    w45 = w45 + (learning_rate * delta_h5 * h5)
    print(f"w45(new)={w45}")
    w36 = w36 + (learning_rate * delta_h6 * h6)
    print(f"w36(new)={w36}")
    w35 = w35 + (learning_rate * delta_h5 * h5)
    print(f"w35(new)={w35}")
    w26 = w26 + (learning_rate * delta_h6 * h6)
    print(f"w26(new)={w26}")
    w25 = w25 + (learning_rate * delta_h5 * h5)
    print(f"w25(new)={w25}")
    w16 = w16 + (learning_rate * delta_h6 * h6)
    print(f"w16(new)={w16}")
    w15 = w15 + (learning_rate * delta_h5 * h5)
    print(f"w15(new)={w15}")

    # New Biases
    t7 = t7 + (learning_rate * delta_o7)
    print(f"t7(new)={t7}")
    t5 = t5 + (learning_rate * delta_h5)
    print(f"t5(new)={t5}")
    t6 = t6 + (learning_rate * delta_h6)
    print(f"t6(new)={t6}")

    epoch += 1
    print("---end of epoch---")
print(f"final y after {epoch} epochs = {o7} with error = {error}")
print(epoch)

