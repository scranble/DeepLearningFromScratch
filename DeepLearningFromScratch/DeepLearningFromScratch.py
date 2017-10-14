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

# multiplier layer
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    # dout / dx = (dout / dz) * (dz / dx)
    def backword(self, dout):
        dx = dout * self.y # dz / dx = d/dx (xy)
        dy = dout * self.x
        return dx, dy

# layer example
apple = 100
apple_num = 2
tax_rate = 1.1

mul_apple_layer = MulLayer() # new object instance
mul_tax_layer = MulLayer()

# forward propergation
def MulLayer_Example(apple, apple_num, mul_apple_layer, mul_tax_layer):
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax_rate) # should be casted

    print(price) # 220
    return price

price = MulLayer_Example(apple, apple_num, mul_apple_layer, mul_tax_layer)

# backward propergation
def MulLayer_Backword_Example(mul_apple_layer, mul_tax_layer):
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backword(dprice)
    dapple, dapple_num = mul_apple_layer.backword(dapple_price)
    print(dapple, dapple_num, dtax) # 2,2 110 200
    return dapple, dapple_num, dtax

dapple, dapple_num, dtax  = MulLayer_Backword_Example(mul_apple_layer, mul_tax_layer)

# adder layer
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
    # dout / dx = (dout / dz) * (dz / dx)
    # dz / dx = d/dx (x + y)
        dx = dout * 1
        dy = dout * 1
        return dx, dy

# example
# ( 100 * 2 + 150 * 3 ) * 1.1
orange = 150
orange_num = 3

mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()

def Mul_And_Add_Forward_Example(apple, apple_num, orange, orange_num, tax_rate,
                                add_apple_orange_layer, 
                                mul_apple_layer,
                                mul_orange_layer,
                                mul_tax_layer):
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax_rate)
    
    print(price)
    return price

price = Mul_And_Add_Forward_Example(apple, apple_num, orange, orange_num, tax_rate,
                                    add_apple_orange_layer,
                                    mul_apple_layer,
                                    mul_orange_layer,
                                    mul_tax_layer)
 # 715

dprice = 1
def Mul_And_Add_BackWard_Example(dprice):
    dall_price, dtax = mul_tax_layer.backword(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dorange, dorange_num = mul_orange_layer.backword(dorange_price)
    dapple, dapple_num = mul_apple_layer.backword(dapple_price)
    print(dapple, dapple_num, dorange, dorange_num, dtax) # 2.2, 110, 3.3, 165, 650
    return dapple, dapple_num, dorange, dorange_num, dtax

