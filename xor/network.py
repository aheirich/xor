#
# xor network with 10 relu units in the hidden layer
#

import numpy as np

input_hidden_weights = [[-0.38688731,  0.26221564, -0.63275653, -0.63185972,  0.56936806,  0.66896194,
   0.49006391,  0.44972926, -0.49830455,  0.5427345 ],
 [-0.14791906,  0.38528341, -0.45161089,  0.6326226,   0.29551026, -0.56385952,
   0.49083051, -0.44947305,  0.10054732,  0.54112089]]

hidden_bias = [  0.00000000e+00,  -9.36306096e-05,   0.00000000e+00,   2.66444185e-05,
   1.97773712e-04,  -3.87232634e-03,  -4.90267992e-01,  -1.47563347e-04,
  -1.17809914e-01,  -5.40929019e-01]

hidden_output_weights = [[ 0.4027527 ],
 [ 0.3184571 ],
 [-0.3466984 ],
 [ 1.06400228],
 [ 0.30103049],
 [ 0.62229204],
 [-1.05087543],
 [ 0.91520125],
 [-0.23849073],
 [-0.83612299]]

output_bias = [-0.32433593]

# compute forward propagation

#
input_activation = [[ 0 ], [ 1 ]]
# z[1] = W[1] x + b[1]
#      = input_hidden_weights input_activation + hidden_bias
# a[1] = g(z[1])

W1 = np.array(input_hidden_weights)
B1 = np.array(hidden_bias)
x = np.array(input_activation)

z1_ = np.matmul(np.transpose(x), W1)
z1 = z1_ + B1
a1 = np.maximum(z1, 0) # relu

print("W1", W1)
print("B1", B1)
print("x", x)
print("z1_", z1_)
print("z1", z1)
print("a1", a1)

# z[2] = W[2] a[1] + B[2]
#      = hidden_output_weights a1 + output_bias

W2 = np.array(hidden_output_weights)
B2 = np.array(output_bias)

z2_ = np.matmul(a1, W2)
z2 = z2_ + B2
a2 = np.maximum(z2, 0)

print("W2", W2)
print("B2", B2)
print("a1", a1)
print("z2_", z2_)
print("z2", z2)
print("a2", a2)

