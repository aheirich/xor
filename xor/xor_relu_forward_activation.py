
#
# xor network with 16 relu units in the hidden layer
#

import math
import numpy
from scipy import optimize


# W1 2 inputs, 10 hidden
input_hidden_weights = [[-0.4055348,  -0.52823716, -0.43952978, -0.18343377, -0.47216558, -0.32241216,
                         -0.55611771, -0.38063344,  0.04775526, -0.76964206, -0.44537184,  0.58646101,
                         0.83039683, -0.26692185, -0.48729387, -0.3081055 ],
                        [-0.29524794,  0.52825111, -0.21107587, -0.23073712, -0.55283833, -0.12979296,
                         0.55611771, -0.36968827,  0.00462923,  0.76966035, -0.15453213, -0.58635885,
                         -0.83039653, -0.04750431, -0.08500406,  0.4753032 ]]

# B1
hidden_bias = [ 2.95232803e-01,  -1.23388522e-09,   0.00000000e+00,   0.00000000e+00,
               0.00000000e+00,   0.00000000e+00,   2.25655539e-09,   0.00000000e+00,
               2.17137873e-01,  -4.03572953e-09,   0.00000000e+00,   3.15902193e-10,
               1.22724542e-09,   0.00000000e+00,   0.00000000e+00,   3.08106124e-01 ]

# W2 10 hidden, 1 output
hidden_output_weights =  [[ 0.3061325,   0.7249217 ],
                          [ 0.82212615, -0.87020439],
                          [-0.16003487,  0.05678433],
                          [ 0.03079337,  0.03488749],
                          [-0.12205628, -0.03029424],
                          [-0.34037149, -0.16318902],
                          [ 0.15318292, -1.06993532],
                          [-0.17671171,  0.4517504 ],
                          [-0.05496955,  0.47440329],
                          [ 0.6847145,  -0.35256037],
                          [-0.3802914,   0.24590087],
                          [ 0.69478261, -0.19410631],
                          [ 0.86003262, -0.41682884],
                          [-0.32823354,  0.22413403],
                          [ 0.57643712, -0.20763171],
                          [ 0.09290985,  1.13164198]]

# B2
output_bias = [ -0.10707041,  0.33430237 ]

W1 = numpy.array(input_hidden_weights)
B1 = numpy.array(hidden_bias)
W2 = numpy.array(hidden_output_weights)
B2 = numpy.array(output_bias)
gX = None
gZ1 = None

print("W1 input_hidden_weights", W1)
print("B1 hidden_bias", B1)
print("W2 hidden_output_weights", W2)
print("B2 output_bias", B2)



# compute forward propagation

def forward_activation(input_activation):
  global gZ1
  global gX
  #
  #input_activation = [[ 1 ], [ 0 ]]
  #input_activation = [[ 0 ], [ 1 ]]
  # z[1] = W[1] x + B[1]
  #      = input_hidden_weights . input_activation + hidden_bias
  # a[1] = g(z[1]), g is Relu
  print("forward_activation", input_activation)
  
  x = numpy.array(input_activation)
  z1_ = numpy.matmul(numpy.transpose(x), W1)
  z1 = z1_ + B1
  a1 = numpy.maximum(z1, 0) # relu
  
  gZ1 = z1
  gX = [[ x[0][0], x[1][0] ]]
  
  print("z1", z1[0])
  print("a1", a1[0])
  
  # z[2] = W[2] a[1] + B[2]
  #      = hidden_output_weights a1 + output_bias
  # a[2] = g(z[2])
  
  z2_ = numpy.matmul(a1, W2)
  z2 = z2_ + B2
  a2 = numpy.maximum(z2, 0)
  
  print("z2", z2[0])
  print("a2", a2[0])
  print("")
  print("**** xor(", input_activation, ") = ", a2[0], "=", z2[0])
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

#cases = [ [[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]] ]
cases = [ [[0.115414], [0.847325]] ]

for case in cases:
  print("")
  print("**** case ", case, " ****")
  print("")
  z2 = forward_activation(case)
#x = backward_activation(z2)
#print("case", case, "x", x)


