#
# xor network with 10 relu units in the hidden layer
#

import numpy
import scipy.optimize

input_hidden_weights = [[-0.38688731,  0.26221564, -0.63275653, -0.63185972,  0.56936806,  0.66896194,
                         0.49006391,  0.44972926, -0.49830455,  0.5427345 ],
                        [-0.14791906,  0.38528341, -0.45161089,  0.6326226,   0.29551026, -0.56385952,
                         0.49083051, -0.44947305,  0.10054732,  0.54112089]]

hidden_bias = [  0.00000000e+00,  -9.36306096e-05,   0.00000000e+00,   2.66444185e-05,
               1.97773712e-04,  -3.87232634e-03,  -4.90267992e-01,  -1.47563347e-04,
               -1.17809914e-01,  -5.40929019e-01]

hidden_output_weights = [[ 0.4027527 ,
                          0.3184571 ,
                          -0.3466984 ,
                          1.06400228,
                          0.30103049,
                          0.62229204,
                          -1.05087543,
                          0.91520125,
                          -0.23849073,
                          -0.83612299]]

output_bias = [-0.32433593]

VERBOSE = False

# compute forward propagation

def forward_activation(input_activation):
#
#input_activation = [[ 1 ], [ 0 ]]
#input_activation = [[ 0 ], [ 1 ]]
# z[1] = W[1] x + B[1]
#      = input_hidden_weights input_activation + hidden_bias
# a[1] = g(z[1])

  W1 = numpy.array(input_hidden_weights)
  B1 = numpy.array(hidden_bias)
  x = numpy.array(input_activation)

  z1_ = numpy.matmul(numpy.transpose(x), W1)
  z1 = z1_ + B1
  a1 = numpy.maximum(z1, 0) # relu

  if VERBOSE:
    print("W1", W1)
    print("B1", B1)
    print("x", x)
    print("z1_", z1_)
    print("z1", z1)
  print("a1", a1)

# z[2] = W[2] a[1] + B[2]
#      = hidden_output_weights a1 + output_bias
# a[2] = g(z[2])

  W2 = numpy.array(hidden_output_weights)
  B2 = numpy.array(output_bias)

  z2_ = numpy.matmul(W2, numpy.transpose(a1))
  z2 = z2_ + B2
  a2 = numpy.maximum(z2, 0)

  if VERBOSE:
    print("W2", W2)
    print("B2", B2)

    print("z2_", z2_)
    print("z2", z2)
  print("a2", a2)
  print("")
  print("**** xor(", x, ") = ", a2)
  print("")
  return a2






def terse_callback(xk, **kwargs):
  print("xk", xk)
  print("kwargs nit", kwargs['nit'])






# compute backward receptive field

def backward_activation(a2):
  output_activation = numpy.array(a2)
  z2 = output_activation

# a[2] = output_activation
# z[2] = g^-1(a[2]) = a[2]
# W[2] a[1] = z[2] - B[2]

# find a[1] by linear programming.  Given W[2], zb2 = z[2] - B[2].
# minimize c^T x subject to A_eq x == b_eq
# pick arbitrary nonzero c, A_eq = W[2], b_eq = (z[2] - B[2])

# c minimize a1[0]
# note we could pick any c here
  #c2 = numpy.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ])
  c2 = numpy.array([ 0, 1, 0, 0, 1, 0, 0, 0, 0, 0 ])

  W2 = numpy.array(hidden_output_weights)
  B2 = numpy.array(output_bias)

  A2_eq = W2
  b2_eq = (z2 - B2)[0]

  if VERBOSE:
    print("c2", c2)
    print("A2_eq", A2_eq)
    print("b2_eq", b2_eq)

  options = dict([('maxiter', 10), ('disp', True)])
  __a1 = scipy.optimize.linprog(c2, A_eq=A2_eq, b_eq=b2_eq, bounds=( 0, 1 ), options=options)
  print("success == ", __a1.success)
  print("x == __a1.x ==", __a1.x)

  a1_ = __a1.x
  a1W2 = a1_[0] * W2[0][0] + a1_[1] * W2[0][1] + a1_[2] * W2[0][2] + a1_[3] * W2[0][3] + a1_[4] * W2[0][4] + a1_[5] * W2[0][5] + a1_[6] * W2[0][6] + a1_[7] * W2[0][7] + a1_[8] * W2[0][8] + a1_[9] * W2[0][9] + B2[0];

  if VERBOSE:
    print("a1 W2 = ", a1W2)
    print("**** does x W2 equal a2? ***")
    error = a1W2 - a2[0][0]
    print("error", error)

  z1 = [a1_]

# z[1] = g^-1(a[1]) = a[1]
# W[1] x = z[1] - B[1]

# find x by algebra
# a[1][0] = x[0] W[1][0][0] + x[1] W[1][1][0]
# a[1][1] = x[0] W[1][0][1] + x[1] W[1][1][1]

# a10 = x0 w100 + x1 w110
# x0 = (a10 - x1 w110) / w100

# a11 = x0 w101 + x1 w111
#    = w101 (a10 - x1 w110) / w100 + x1 w111
#    = (w101 / w100) (a10 - x1 w110) + x1 w111
#    = k1 a10 - k1 x1 w110 + x1 w111
#    = k1 a10 + x1 (w111 - k1 w110)
# (a11 - k1 a10) / (w111 - k1 w110) = x1

  W1 = numpy.array(input_hidden_weights)
  B1 = numpy.array(hidden_bias)

  j0 = 0
  j1 = 1
  k1 = W1[0][j1] / W1[0][j0]
  z1B1 = z1 - B1
  x1_ = (z1B1[0][j1] - k1 * z1B1[0][j0]) / (W1[1][j1] - k1 * W1[1][j0])
  x0_ = (z1B1[0][j0] - x1_ * W1[1][j0]) / W1[0][j0]
  x_ = numpy.array([ [x0_], [x1_] ])
  return x_


# there are two cases: fan-in, or fan-out, traversing from input to output.
# fan-out: when tracing backwards there are more constraints than unknowns.
# solve the problem algebraically.
# fan-in: there are more unknowns than constraints.  use linear programming
# with equality constraints to get a feasible solution.  apply systematic
# perturbations in the vicinity of this solution to sample other solutions.
# use the weights to find perturbations that add zero to the dot product.

cases = [ [[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]] ]

for case in cases:
  print("")
  print("**** case ", case, " ****")
  print("")
  a2 = forward_activation(case)
  print("**** a2", a2)
  x = backward_activation(a2)
  print("")
  print("**** forward a2", a2)
  print("**** backward x", x)









