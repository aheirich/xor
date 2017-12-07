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

# compute forward propagation

#
input_activation = [[ 1 ], [ 0 ]]
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

print("W2", W2)
print("B2", B2)

print("z2_", z2_)
print("z2", z2)
print("a2", a2)
print("")
print("xor(", x, ") = ", a2)
print("")

# compute backward receptive field

output_activation = numpy.array([ 1 ])

# a[2] = output_activation
# z[2] = g^-1(a[2]) = a[2]
# W[2] a[1] = z[2] - B[2]

# find a[1] by linear programming.  Given W[2], zb2 = z[2] - B[2].
# minimize c^T x subject to A_eq x == b_eq
# pick arbitrary nonzero c, A_eq = W[2], b_eq = (z[2] - B[2])

# c minimize a1[0]
c = numpy.array([ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])

A2_eq = W2
b2_eq = (z2 - B2)[0]

print("c", c)
print("A2_eq", A2_eq)
print("b2_eq", b2_eq)

__a1 = scipy.optimize.linprog(c, A_eq=A2_eq, b_eq=b2_eq, bounds=( 0, 1 ))

print("__a1", __a1)

xW2 = __a1.x[0] * W2[0][0] + __a1.x[1] * W2[0][1] + __a1.x[2] * W2[0][2] + __a1.x[3] * W2[0][3] + __a1.x[4] * W2[0][4] + __a1.x[5] * W2[0][5] + __a1.x[6] * W2[0][6] + __a1.x[7] * W2[0][7] + __a1.x[8] * W2[0][8] + __a1.x[9] * W2[0][9] + B2[0];
print("x W2 = ", xW2)
print("**** does x W2 equal a2? ***")
error = xW2 - a2[0][0]
print("error", error)

# z[1] = g^-1(a[1]) = a[1]
# W[1] x = z[1] - B[1]
#      x = W[1]^-1 ( z[1] - B[1] )

# find x by linear programming.  Given W[1], zb1 = z[1] - B[1].




