# print("Hello Python World!!")

def AND_simple(x1, x2):
    w1,w2,theta = 0.5, 0.5, 0.5
    temp = x1*w1+x2*w2
    if temp <= theta:
        return 0
    elif temp > theta:
        return 1

# print(AND_simple(0,0)) # 0
# print(AND_simple(0,1)) # 0
# print(AND_simple(1,0)) # 0
# print(AND_simple(1,1)) # 1

import numpy as np

def AND_vector(x1,x2):
    x = np.array([x1,x2])    # input
    w = np.array([0.5,0.5])  # weight
    b = -0.7                 # bias
    temp = np.sum(x*w)+b
    if temp <= 0:
        return 0
    else:
        return 1

def NAND_vector(x1,x2):
    x = np.array([x1,x2])       # input
    w = np.array([-0.5,-0.5])   # weight
    b = 0.7                     # bias
    temp = np.sum(x*w)+b
    if temp <= 0:
        return 0
    else:
        return 1

def OR_vector(x1,x2):
    x = np.array([x1,x2])    # input
    w = np.array([0.5,0.5])  # weight
    b = -0.2                  # bias
    temp = np.sum(x*w)+b
    if temp <= 0:
        return 0
    else:
        return 1

def XOR_vector(x1,x2):
    s1 = NAND_vector(x1,x2)
    s2 = OR_vector(x1,x2)
    y = AND_vector(s1,s2)
    return y

# print(XOR_vector(0,0)) # 0
# print(XOR_vector(0,1)) # 1
# print(XOR_vector(1,0)) # 1
# print(XOR_vector(1,1)) # 0

def step_function(x):
    return np.array( (x > 0), dtype=np.int)

import matplotlib.pylab as plt

#x = np.arange(-5.0, 5.0, 0.1) # like MATLAB/Scilab
#y = step_function(x)
#plt.plot(x,y)
#plt.ylim(-0.1, 1.1)
#plt.show()

def sigmoid(x):
    return 1/ (1+np.exp(-x))

#x = np.arange(-5.0, 5.0, 0.1) # like MATLAB/Scilab
#y = sigmoid(x)
#plt.plot(x,y)
#plt.ylim(-0.1, 1.1)
#plt.show()

def relu(x):
    return np.maximum(0, x)

# Network as Matrix
# X = np.array([1,2])            # input
# W = np.array([[1,3,5],[2,4,6]])  # weight
# Y = np.dot(X,W)
# print(Y)

# 3-layer Network as Matrices
def identity_function(x):
    return x
#layer 1
X = np.array([1.0, 0.5]) # input
W1 = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]]) # weight 1
B1 = np.array([0.1, 0.2, 0.3]) # bias 1
A1 = np.dot(X, W1)+B1
print(A1) # [0.3, 0.7, 1.1]
Z1 = sigmoid(A1)
print(Z1)
#layer 2
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]]) # weight 2
B2 = np.array([0.1,0.2])
A2 = np.dot(Z1,W2)+B2
print(A2)
Z2 = sigmoid(A2)
print(Z2)
#layer 3(out)
W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])
A3 = np.dot(Z2,W3)+B3
print(A3)
Y = identity_function(A3)
print(Y)