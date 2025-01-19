#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1-x)

x1, x2 = 0.35, 0.9
y = 0.5

w13, w14 = 0.1, 0.4
w23, w24 = 0.8, 0.6
w35, w45 = 0.3, 0.9

learning_rate = 1.0

error = float('inf')
epoch = 0
while epoch <= 6:
    print("---new epoch---")
    y3 = w13 * x1 + w23 * x2
    print(f"y3 = {y3}")
    y4 = w14 * x1 + w24 * x2
    print(f"y4 = {y4}")
    h3 = sigmoid(y3)
    print(f"h3= {h3}")
    h4 = sigmoid(y4)
    print(f"h4= {h4}")
    
    z5 = h3 * w35 + h4 * w45
    print(f"z5={z5}")
    o5 = sigmoid(z5)
    error = y - o5
    print(f"o5={o5}, error={error}")

    delta_o5 = error * sigmoid_derivative(o5)

    error_h3 = delta_o5 * w35
    error_h4 = delta_o5 * w45
    delta_h3 = error_h3 * sigmoid_derivative(h3)
    delta_h4 = error_h4 * sigmoid_derivative(h4)

    w35 = w35 + (learning_rate * delta_o5 * h3)
    print(f"w35(new)={w35}")
    w45 = w45 + (learning_rate * delta_o5 * h4)
    print(f"w45(new)={w45}")
    w13 = w13 + (learning_rate * delta_h3 * x1)
    print(f"w13(new)={w13}")
    w23 = w23 + (learning_rate * delta_h3 * x2)
    print(f"w23(new)={w23}")
    w14 = w14 + (learning_rate * delta_h4 * x1)
    print(f"w14(new)={w14}")
    w24 = w24 + (learning_rate * delta_h4 * x2)
    print(f"w24(new)={w24}")

    epoch += 1
    print("---end of epoch---")
print(f"final y after 6 epochs = {o5} with error = {error}")
print(epoch)


# In[ ]:




