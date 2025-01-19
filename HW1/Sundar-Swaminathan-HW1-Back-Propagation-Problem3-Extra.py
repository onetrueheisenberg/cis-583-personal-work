#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1-x)

x1, x2, x3 = 1.0, 0.0, 1.0
t4, t5, t6 = -0.4, 0.2, 0.1
y = 1.0

w14, w15 = 0.2, -0.3
w24, w25 = 0.4, 0.1
w34, w35 = -0.5, 0.2
w46, w56 = -0.3, -0.2

learning_rate = 0.9

error = float('inf')
epoch = 0
while epoch <= 3:
    print("---new epoch---")
    # Calculate Hidden outputs
    y4 = w14 * x1 + w24 * x2 + w34 * x3 + t4
    print(f"y4 = {y4}")
    y5 = w15 * x1 + w25 * x2 + w35 * x3 + t5
    print(f"y5 = {y5}")
    h4 = sigmoid(y4)
    print(f"h4 = {h4}")
    h5 = sigmoid(y5)
    print(f"h5 = {h5}")

    # Calculate Network output
    z6 = h4 * w46 + h5 * w56 + t6
    print(f"z6 = {z6}")
    o6 = sigmoid(z6)
    error = y - o6
    print(f"o6 = {o6}, error = {error}")

    # Calcualte Error
    delta_o6 = error * sigmoid_derivative(o6)

    # Backpropagate Error
    error_h4 = delta_o6 * w46
    error_h5 = delta_o6 * w56
    delta_h4 = error_h4 * sigmoid_derivative(h4)
    delta_h5 = error_h5 * sigmoid_derivative(h5)

    # New weights
    w46 = w46 + (learning_rate * delta_o6 * h4)
    print(f"w46(new)={w46}")
    w56 = w56 + (learning_rate * delta_o6 * h5)
    print(f"w56(new)={w56}")
    w34 = w34 + (learning_rate * delta_h4 * h5)
    print(f"w34(new)={w34}")
    w35 = w35 + (learning_rate * delta_h5 * h4)
    print(f"w35(new)={w35}")
    w24 = w24 + (learning_rate * delta_h4 * x1)
    print(f"w24(new)={w24}")
    w25 = w25 + (learning_rate * delta_h5 * x2)
    print(f"w25(new)={w23}")
    w14 = w14 + (learning_rate * delta_h4 * x1)
    print(f"w14(new)={w14}")
    w15 = w15 + (learning_rate * delta_h5 * x2)
    print(f"w15(new)={w15}")

    # New Biases
    t6 = t6 + (learning_rate * delta_o6)
    print(f"t6(new)={t6}")
    t5 = t5 + (learning_rate * delta_h5)
    print(f"t5(new)={t5}")
    t4 = t4 + (learning_rate * delta_h4)
    print(f"t4(new)={t4}")

    epoch += 1
    print("---end of epoch---")
print(f"final y after 6 epochs = {o6} with error = {error}")
print(epoch)


# In[ ]:




