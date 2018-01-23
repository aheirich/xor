
#
# xor network with 10 relu units in the hidden layer
#

import math
import numpy
from scipy import optimize


# W1 2 inputs, 10 hidden
input_hidden_weights = [[-0.38688731,  0.26221564, -0.63275653, -0.63185972,  0.56936806,  0.66896194,
                         0.49006391,  0.44972926, -0.49830455,  0.5427345 ],
                        [-0.14791906,  0.38528341, -0.45161089,  0.6326226,   0.29551026, -0.56385952,
                         0.49083051, -0.44947305,  0.10054732,  0.54112089]]


# B1
hidden_bias = [ 0.00000000e+00, -9.36306096e-05, 0.00000000e+00,
               2.66444185e-05, 1.97773712e-04, -3.87232634e-03,
               -4.90267992e-01, -1.47563347e-04, -1.17809914e-01,
               -5.40929019e-01 ]

# W2 10 hidden, 1 output
hidden_output_weights = [[ 0.4027527], [  0.3184571], [  -0.3466984], [ 1.06400228],
                         [ 0.30103049], [ 0.62229204], [ -1.05087543], [ 0.91520125],
                         [ -0.23849073], [ -0.83612299 ]]

# B2
output_bias = [ -0.32433593 ]

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

cases = [ [[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]] ]

for case in cases:
  print("")
  print("**** case ", case, " ****")
  print("")
  z2 = forward_activation(case)
  x = backward_activation(z2)
#print("case", case, "x", x)

