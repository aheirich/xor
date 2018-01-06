#
# xor network with 10 relu units in the hidden layer
#

import math
import numpy
from scipy import optimize

# W1
input_hidden_weights = [[-0.38688731,  0.26221564, -0.63275653, -0.63185972,  0.56936806,  0.66896194,
                         0.49006391,  0.44972926, -0.49830455,  0.5427345 ],
                        [-0.14791906,  0.38528341, -0.45161089,  0.6326226,   0.29551026, -0.56385952,
                         0.49083051, -0.44947305,  0.10054732,  0.54112089]]
i = [ [-0.38688731, -0.14791906],
     [0.26221564, 0.38528341],
     [-0.63275653, -0.45161089],
     [-0.63185972, 0.6326226],
     [0.56936806, 0.29551026],
     [0.66896194, -0.56385952],
     [0.49006391, 0.49083051],
     [0.44972926, -0.44947305],
     [-0.49830455, 0.10054732],
     [0.5427345, 0.54112089] ]

# B1
hidden_bias = [  0.00000000e+00,  -9.36306096e-05,   0.00000000e+00,   2.66444185e-05,
               1.97773712e-04,  -3.87232634e-03,  -4.90267992e-01,  -1.47563347e-04,
               -1.17809914e-01,  -5.40929019e-01]

# W2
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

# B2
output_bias = [-0.32433593]

print("input_hidden_weights", input_hidden_weights)
print("hidden_bias", hidden_bias)
print("hidden_output_weights", hidden_output_weights)
print("output_bias", output_bias)

VERBOSE = True

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

  print("forward z1", z1)

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
  #print("kwargs nit", kwargs['nit'])


def ratio(x):
  scale = 1000000
  return str(int(x * scale)) + "/" + str(scale)


def write_lrs_file(case, A, b):
  casename = "xor_" + str(case[0][0]) + "_" + str(case[1][0])
  file = open(casename + ".ine", "w")
  file.write(casename + "\n")
  file.write("H-representation\n")
  file.write("linearity " + str(len(b)))
  for i in range(len(b)):
    file.write(" " + str(i + 1))
  file.write("\n")
  file.write("begin\n")
  file.write("1 " + str(len(A[0]) + 1) + " rational\n")
  file.write(str(ratio(-b[0])))
  for i in range(len(A[0])):
    file.write(" " + str(ratio(A[0][i])))
  file.write("\n")
  file.write("end\n")
  file.close()



# compute backward receptive field

def backward_activation(case, a2):
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
    print("z2", z2)
    print("B2", B2)
    print("c2", c2)
    print("A2_eq", A2_eq)
    print("b2_eq", b2_eq)

  #write_lrs_file(case, A2_eq, b2_eq)
  #options = dict([('maxiter', 10), ('disp', True)])
  #__a1 = optimize.linprog(c2, A_eq=A2_eq, b_eq=b2_eq, bounds=( 0, 1 ), options=options)
  #print("success == ", __a1.success)
  #print("x == __a1.x ==", __a1.x)

  #a1_ = __a1.x


  # need to sample all possible values of underdetermined system of linear equations
  # one equation in ten unknowns
  # bound the max value of each unknown
  # then sample monte carlo
  # a 10-D space is a lot of samples, concentrate around the zero point in each axis that's where the nonlinearity occurs
  # a. bounding the max values
  # we know that W.a1 + b = 0 so W.a1 = -b
  # extreme value of a1[i] is at W[i]a1[1] = -b or a1[1] = -b/W[i]
  # so sample a1[i] from 0 through -b/W[i] using logarithmic spacing
  # make this dynamic, select the range for a1[i] assuming all other a1[j] are fixed for j < i, this can reduce the range for a1[i] if a1[j] > 0
  # sample in ten nested loops, each loop sets a range and chooses k sample points using log spacing



#  a1W2 = a1_[0] * W2[0][0] + a1_[1] * W2[0][1] + a1_[2] * W2[0][2] + a1_[3] * W2[0][3] + a1_[4] * W2[0][4] + a1_[5] * W2[0][5] + a1_[6] * W2[0][6] + a1_[7] * W2[0][7] + a1_[8] * W2[0][8] + a1_[9] * W2[0][9] + B2[0];
#
#  if VERBOSE:
#    print("a1 W2 = ", a1W2)
#    print("**** does x W2 equal a2? ***")
#    error = a1W2 - a2[0][0]
#    print("error", error)
#
#  z1 = [a1_]

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

#  W1 = numpy.array(input_hidden_weights)
#  B1 = numpy.array(hidden_bias)
#
#  j0 = 0
#  j1 = 1
#  k1 = W1[0][j1] / W1[0][j0]
#  z1B1 = z1 - B1
#  x1_ = (z1B1[0][j1] - k1 * z1B1[0][j0]) / (W1[1][j1] - k1 * W1[1][j0])
#  x0_ = (z1B1[0][j0] - x1_ * W1[1][j0]) / W1[0][j0]
#  x_ = numpy.array([ [x0_], [x1_] ])
#  return x_
  return [ [0], [0] ]


def extremum(partialSum, w, b, z2):
  # w.a1 + b + partialSum = z2
  # w.a1 = z2 - partialSum - b
  # a1 = (z2 - partialSum - b) / w
  result = (z2 - b - partialSum) / w
  return result


def samples(maxRange):
  return [ 0, maxRange * 0.02, maxRange ]

#('**** a2', array([[ 0.]]))
#('z2 samples', [-0.0, -0.12082581000000001, -0.24165162000000001, -0.36247743000000004, -0.48330324000000002, -0.60412905000000006, -0.72495486000000009, -0.84578067000000012, -0.96660648000000005, -1.0874322900000002, -1.2082581000000001])
#('sample z2', -0.0)
#([0, 0, 0, 0, 0, 0, 0, 0, 0, -0.38790457131193101], 'error', -0.0)

#('**** a2', array([[ 0.5597363]]))
#('z2 samples', [array([[ 0.5597363]])])
#('sample z2', array([[ 0.5597363]]))
#([0, 0, 0, 0, 0, 0, 0, 0, 0, array([[-1.05734711]])], 'error', array([[  2.22044605e-16]]))


def maxZ(W):
  sumWeights = 0
  for w in W:
    sumWeights = sumWeights + w[0]
  maxActivation = 3
  return maxActivation * sumWeights

def makeSamples(z, W):
  print("makeSamples z", z, "W", W)
  if z > 0:
    return z[0]
  result = []
  endpoint = maxZ(W)
  print("endpoint", endpoint)
  numSamples = 11
  for i in range(numSamples):
    result.append(i * -endpoint / (numSamples - 1))
  return result

def nonsense(x):
  if x < -0.25 or x > 1.25:
    return True
  if x > 0.25 and x < 0.75:
    return True
  return False

inconsistent_solutions = 0
nonsense_inputs = 0
reasonable_inputs = 0
epsilon = 0.001

def computeInputLayer(a1):
  global inconsistent_solutions
  global nonsense_inputs
  global reasonable_inputs
  # layer 1 to input is overdetermined, solve by elimination
  W1 = numpy.array(input_hidden_weights)
  B1 = numpy.array(hidden_bias)
  z1 = a1
  # TODO explore z1[j]<0
  # find a solution for the first two unknowns/equations
  j0 = 0
  j1 = 1
  k1 = W1[0][j1] / W1[0][j0]
  z1B1 = z1 - B1
  x1_ = (z1B1[j1] - k1 * z1B1[j0]) / (W1[1][j1] - k1 * W1[1][j0])
  x0_ = (z1B1[j0] - x1_ * W1[1][j0]) / W1[0][j0]
  # test this solution for consistency with remaining equations
  consistent = True
  for j in range(10):
    # W1[0][j] x0_ + W1[1][j] x1_ + B1[j] = z1[j]
    z1_j = W1[0][j] * x0_ + W1[1][j] * x1_ + B1[j]
    error = math.fabs(z1_j - z1[j])
    reconstruction_error_threshold = 0.01
    if error > reconstruction_error_threshold:
      print("inconsistent solution, error =", error, a1)
      inconsistent_solutions = inconsistent_solutions + 1
      consistent = False
      break
  if consistent:
    # test x0_, x1_ for statistical reasonableness
    if nonsense(x0_) or nonsense(x1_):
      print("nonsense input", x0_, x1_)
      nonsense_inputs = nonsense_inputs + 1
    else:
      print("reasonable input", x0_, x1_)
      reasonable_inputs = reasonable_inputs + 1



def backward_sampling(a2):

  # layer 2 to layer 1 is underdetermined, sample the free variables
  W2 = numpy.array(hidden_output_weights)
  B2 = numpy.array(output_bias)
  z2 = a2
  z2Samples = makeSamples(z2, W2)
  print("z2", z2, "z2 samples", z2Samples)
  
  for z2 in z2Samples:
    print("sample z2", z2)
    numSamples = 3

    max0 = extremum(0, W2[0][0], B2[0], z2)
    sample0 = samples(max0)
    lastSample = []
  
    for i0 in range(numSamples):
      partial0 = W2[0][0] * sample0[i0]
      max1 = extremum(partial0, W2[0][1], B2[0], z2)
      sample1 = samples(max1)

      for i1 in range(numSamples):
        partial1 = partial0 + W2[0][1] * sample1[i1]
        max2 = extremum(partial1, W2[0][2], B2[0], z2)
        sample2 = samples(max2)

        for i2 in range(numSamples):
          partial2 = partial1 + W2[0][2] * sample2[i2]
          max3 = extremum(partial2, W2[0][3], B2[0], z2)
          sample3 = samples(max3)

          for i3 in range(numSamples):
            partial3 = partial2 + W2[0][3] * sample3[i3]
            max4 = extremum(partial3, W2[0][4], B2[0], z2)
            sample4 = samples(max4)

            for i4 in range(numSamples):
              partial4 = partial3 + W2[0][4] * sample4[i4]
              max5 = extremum(partial4, W2[0][5], B2[0], z2)
              sample5 = samples(max5)

              for i5 in range(numSamples):
                partial5 = partial4 + W2[0][5] * sample5[i5]
                max6 = extremum(partial5, W2[0][6], B2[0], z2)
                sample6 = samples(max6)

                for i6 in range(numSamples):
                  partial6 = partial5 + W2[0][6] * sample6[i6]
                  max7 = extremum(partial6, W2[0][7], B2[0], z2)
                  sample7 = samples(max7)

                  for i7 in range(numSamples):
                    partial7 = partial6 + W2[0][7] * sample7[i7]
                    max8 = extremum(partial7, W2[0][8], B2[0], z2)
                    sample8 = samples(max8)

                    for i8 in range(numSamples):
                      partial8 = partial7 + W2[0][8] * sample8[i8]
                      max9 = extremum(partial8, W2[0][9], B2[0], z2)

                      s0 = (sample0[i0])
                      s1 = (sample1[i1])
                      s2 = (sample2[i2])
                      s3 = (sample3[i3])
                      s4 = (sample4[i4])
                      s5 = (sample5[i5])
                      s6 = (sample6[i6])
                      s7 = (sample7[i7])
                      s8 = (sample8[i8])
                      s9 = max9
                      
                      sample = [ s0, s1, s2, s3, s4, s5, s6, s7, s8, s9 ]
                      if lastSample != sample:

                        dotproduct = 0
                        for i in range(10):
                          dotproduct = dotproduct + W2[0][i] * sample[i]
                        W2dotSamplePlusB2 = dotproduct + B2[0]
                        error = z2 - W2dotSamplePlusB2
                        print(sample, "error", error)
                        assert(math.fabs(error) < epsilon)
                        lastSample = sample
                        computeInputLayer(sample)




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
  #x = backward_activation(case, a2)
  backward_sampling(a2)
  print("case", case, "inconsistent solutions", inconsistent_solutions, "nonsense inputs", nonsense_inputs, "reasonable inputs", reasonable_inputs)
  print("")









