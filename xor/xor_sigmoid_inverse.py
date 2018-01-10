
#
# xor network with 10 sigmoid units in the hidden layer
#

import math
import numpy
from scipy import optimize

# 2 inputs, 2 hidden, 2 outputs

#input weights to hidden layer
W1 = numpy.array([[ 4.76426458,  3.15521908],
      [ 4.49576712,  3.08746362]])
#hidden layer biases
B1 = numpy.array([-1.84902346, -4.64604139])
#hidden weights to output layer
W2 = numpy.array([[ 4.64140081, -5.0876627 ],
      [-4.76921082,  5.23339081]])
#output layer biases
B2 = numpy.array([-2.06331062,  2.27055311])



print("W1 input_hidden_weights", W1)
print("B1 hidden_bias", B1)
print("W2 hidden_output_weights", W2)
print("B2 output_bias", B2)


def g(x):
  result = []
  for x_ in x:
    result.append(1.0 / (1.0 + math.exp(-x_)))
  return result

def gInverse(x):
  result = []
  for x_ in x:
    result.append(math.log(x_ / (1.0 - x_)))
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
  print("B1", B1)
  z1 = (z1_ + B1)[0]
  print("z1", z1)
  a1 = [g(z1)]
  print("a1", a1)
  
  # z[2] = W[2] a[1] + B[2]
  #      = hidden_output_weights a1 + output_bias
  # a[2] = g(z[2])
  
  z2_ = numpy.matmul(a1, W2)
  print("W2.a1", z2_)
  z2 = (z2_ + B2)[0]
  print("z2", z2)
  a2 = g(z2)
  print("a2", a2)
  print("")
  print("**** xor(", input_activation, ") = ", a2, "= g(", z2, ")")
  print("")
  return (a1, a2, x)


# Solve a Newton system:
# W2.a1 + B2 - z2 = 0
# W1.x + B1 - z1 = 0
# eliminate a1
# W2 . logistic(z1) + B2 - z2 = 0
# W1 . x + B1 - z1 = 0
def F(x, *args):
  z2 = args[0]
  z10 = x[0]
  z11 = x[1]
  x0 = x[2]
  x1 = x[3]
  return [ W2[0][0] * (1/(1+math.exp(-z10))) + W2[1][0] * (1/(1+math.exp(-z11))) + B2[0] - z2[0], \
          W2[0][1] * (1/(1+math.exp(-z10))) + W2[1][1] * (1/(1+math.exp(-z11))) + B2[1] - z2[1], \
          W1[0][0] * x0 + W1[1][0] * x1 + B1[0] - z10, \
          W1[0][1] * x0 + W1[1][1] * x1 + B1[1] - z11 ]


# Jacobian derived by Wolfram Alpha
def DF(x, *args):
  a = x[0]
  b = x[1]
  return \
  [ [ (W2[0][0] * math.exp(-a))/((1 + math.exp(-a)) * (1 + math.exp(-a))), \
   (W2[1][0] * math.exp(-b))/((1 + math.exp(-b)) * (1 + math.exp(-b))), \
   0, \
   0 ], \
 [ (W2[0][1] * math.exp(-a))/((1 + math.exp(-a)) * (1 + math.exp(-a))), \
 (W2[1][1] * math.exp(-b))/((1 + math.exp(-b))  * (1 + math.exp(-b))), \
   0, \
   0 ], \
 [ -1, 0, W1[0][0], W1[1][0] ], \
 [ 0, -1, W1[0][1], W1[1][1] ] ]



def backward_activation(a1Forward, a2, xForward):
  print("backward activation(", a2, ")")
  z2 = gInverse(a2)
  print("z2", z2)
  xInitial = numpy.array([ 0, 0, 0, 0 ])
  solution = optimize.fsolve(F, xInitial, z2, fprime=DF)
  return solution




##########################################################

cases = [ [[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]] ]

for case in cases:
  print("")
  print("**** case ", case, " ****")
  print("")
  (a1, a2, x) = forward_activation(case)
  solution = backward_activation(a1, a2, x)
  xSolution = solution[2:]
  print("case", case, "inverted solution", xSolution)


