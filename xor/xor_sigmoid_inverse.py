
#
# xor network with 10 relu units in the hidden layer
#

import math
import numpy
from scipy import optimize


# W1 2 inputs, 5 hidden
W1 = [[-3.66874623,  3.96872973,  4.58811092, -3.42505884,  3.931144  ],
      [-4.57925892,  4.58775187, -6.00651932, -4.14732027,  1.4203583 ]]

B1 = [ 0.57311267, -0.66280091, -2.59247088,  0.26248473, -3.34845424]

W2 = [[-2.91450429],
      [ 2.25845695],
      [ 3.54164791],
      [-2.85040522],
      [-4.06980085]]

B2 = [-0.11827146]

print("W1 input_hidden_weights", W1)
print("B1 hidden_bias", B1)
print("W2 hidden_output_weights", W2)
print("B2 output_bias", B2)


def g(x):
  print("g(", x, ")")
  result = []
  for x_ in x:
    result.append(1.0 / (1.0 + math.exp(-x_)))
  return result

def gInverse(x):
  result = []
  for x_ in x:
    result.append(gInverse(math.log(x / (1.0 - x_))))
  return result

# compute forward propagation

def forward_activation(input_activation):
  #
  #input_activation = [[ 1 ], [ 0 ]]
  #input_activation = [[ 0 ], [ 1 ]]
  # z[1] = W[1] x + B[1]
  #      = input_hidden_weights . input_activation + hidden_bias
  # a[1] = g(z[1]), g is Relu
  print("forward_activation", input_activation)
  
  x = numpy.array(input_activation)
  z1_ = numpy.matmul(x, W1)
  print("W1.x", z1_)
  z1 = (z1_ + B1)[0]
  print("z1", z1)
  a1 = g(z1)
  print("a1", a1)
  
  # z[2] = W[2] a[1] + B[2]
  #      = hidden_output_weights a1 + output_bias
  # a[2] = g(z[2])
  
  z2_ = numpy.matmul(a1, W2)
  print("W2.a1", z2_)
  z2 = (z2_ + B2)
  print("z2", z2)
  a2 = g(z2)
  print("a2", a2)
  print("")
  print("**** xor(", input_activation, ") = ", a2, "=", z2)
  print("")
  return z2[0]


# TODO deal with a2==0



def backward_activation(z2):
  print("backward activation(", z2, ")")
  #
  # output to hidden layer
  #
#
# linear program
# W2.a1 + B2 = z2
# W1.x + B1 = z1
#
# W2.a1 - z2 = -B2
# W1.x - z1 = -B1
# a1 = g(z1) = relu(z1)
# unknowns are a1[10], x[2]
# B is -B2 concat -B1 (1 concat 10)
# top row of A is W2.T
# next ten rows are zeroes except diagonals are -1 (for -z1)
# bottom right corner is 2x10 weights W1
#
  _topLeft = W2.T
  _bottomLeft = numpy.zeros((10, 10))
  for i in range(10):
    _bottomLeft[i, i] = -1
  _topRight = numpy.array([[ 0, 0 ]])
  _bottomRight = W1.T
  _top = numpy.concatenate((_topLeft, _topRight), axis=1)
  _bottom = numpy.concatenate((_bottomLeft, _bottomRight), axis=1)
  A_eq = numpy.concatenate((_top, _bottom), axis=0)
  print("A_eq", A_eq)
  c = numpy.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ])
  b_eq = numpy.concatenate((-B2, -B1.T), axis=0)
  print("b_eq", b_eq)
  result = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq)
  print("linprog result", result)
# problem this is a nonsquare matrix we are missing one constraint
# linprog cannot find a feasible solution
# add one more constract z2=...
# now it is a square matrix solve by elimination
#
  print("gZ1", gZ1)
  print("gX", gX)
  u = numpy.concatenate((gZ1, gX), axis=1)
  print("u", u)
  b = numpy.matmul(A_eq, u.T)
  print("b", b)
  print("b_eq", b_eq)






##########################################################

cases = [ [[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]] ]

for case in cases:
  print("")
  print("**** case ", case, " ****")
  print("")
  z2 = forward_activation(case)
#x = backward_activation(z2)
#print("case", case, "x", x)


