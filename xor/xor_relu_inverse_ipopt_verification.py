#
# verify a solution produced by the NLP solver
#
import math
import numpy

# cut and paste the solution here

x = []
for i in range(38):
  x.append(0)

x[0] = 2.499838e+00
x[1] = 1.760863e+00
x[2] = 2.409611e+00
x[3] = 2.482001e+00
x[4] = 2.259150e+00
x[5] = 9.853600e-01
x[6] = 2.524106e+00
x[7] = 1.649710e+00
x[8] = 2.613006e+00
x[9] = 2.012143e+00
x[10] = 4.774024e+00
x[11] = 3.616278e+00
x[12] = 2.271486e+00
x[13] = -6.132143e-01
x[14] = -1.247891e+00
x[15] = 1.347907e+00
x[16] = 2.460233e+00
x[17] = 1.705945e+00
x[18] = -4.996765e+00
x[19] = -2.981081e-01
x[20] = 1.000000e+00
x[21] = 1.000000e+00
x[22] = 1.000000e+00
x[23] = 1.000000e+00
x[24] = 1.000000e+00
x[25] = 1.000000e+00
x[26] = 1.000000e+00
x[27] = 1.000000e+00
x[28] = 1.000000e+00
x[29] = 1.000000e+00
x[30] = -1.206045e-17
x[31] = 1.000000e+00
x[32] = 1.000000e+00
x[33] = -2.629307e-12
x[34] = 7.274952e-12
x[35] = 1.000000e+00
x[36] = 1.000000e+00
x[37] = 1.000000e+00


numOutputUnits = 2
numHiddenUnits = 16
numInputUnits = 2
numActivations = numOutputUnits + numHiddenUnits + numInputUnits;

z2 = numpy.array(x[0 : numOutputUnits])
z1 = numpy.array(x[numOutputUnits : numOutputUnits + numHiddenUnits])
z0 = numpy.array(x[numOutputUnits + numHiddenUnits : numActivations])
alpha2 = numpy.array(x[numActivations : numActivations + numOutputUnits])
alpha1 = numpy.array(x[numActivations + numOutputUnits : numActivations + numOutputUnits + numHiddenUnits])


use_initial_conditions = False
if use_initial_conditions:
  ### initial z, alpha
  z2 = numpy.array([1, 1.19367e-07])
  z1 = numpy.array([-0.110302, -0.528237, -0.43953, -0.183434, -0.472166, -0.322412, -0.556118, -0.380633, 0.264893, -0.769642, -0.445372, 0.586461, 0.830397, -0.266922, -0.487294, 6.24e-07])
  z0 = numpy.array([1, 0])




print "Z2:", z2
print "Z1:", z1
print "Z0:", z0
print "alpha2", alpha2
print "alpha1", alpha1



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

print "W2:", W2
print "B2:", B2
print "W1:", W1
print "B1:", B1


# compute forward activation

z1_ = numpy.matmul(numpy.transpose(z0), W1)
zz1 = z1_ + B1
a1 = numpy.maximum(zz1, 0) # relu
z2_ = numpy.matmul(a1, W2)
zz2 = z2_ + B2
a2 = numpy.maximum(zz2, 0)

print "a1:", a1
print "a2:", a2

print "does a2 == z2?"
print z2

