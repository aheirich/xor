#
# XOR Relu network
#

param l0; # width of input layer 0
param l1; # width of hidden layer 1
param l2; # width of output layer 2

param W1{i in 1..l0, j in 1..l1}; # weights from input i to hidden layer j
param W2{i in 1..l1, j in 1..l2}; # weights from hidden i to output layer j

param b1{i in 1..l1}; # biases hidden layer
param b2{i in 1..l2}; # biases output layer

param y_target{i in 1..l2};

# preactivations
var z2{i in 1..l2};
var z1{i in 1..l1};
var z0{i in 1..l0};

# activations
var a2{i in 1..l2};
var a1{i in 1..l1};


# Relu(x) = x * (tanh(100 * x) + 1) / 2 differentiable form
# TODO find max value between 100 and 1000 to multiply x without overflow

# Objective is distance to target y
minimize loss: sum{i in 1..l2}(y_target[i] - a2[i])^2;

# variable ranges
subject to rangemax2{i in 1..l2}: z2[i] <= 1;
subject to rangemin2{i in 1..l2}: z2[i] >= -1;

subject to rangemax1{i in 1..l1}: z1[i] <= 1;
subject to rangemin1{i in 1..l1}: z1[i] >= -1;

subject to rangemax0{i in 1..l0}: z0[i] <= 1;
subject to rangemin0{i in 1..l0}: z0[i] >= 0;


# activations from preactivations
subject to activation2{i in 1..l2}:
a2[i] = z2[i] * (tanh(100 * z2[i]) + 1) * 0.5;

subject to activation1{i in 1..l1}:
a1[i] = z1[i] * (tanh(100 * z1[i]) + 1) * 0.5;


# constraints for layer 2
subject to zassign2{i in 1..l2}:
z2[i] = sum{j in 1..l1} (W2[j,i] * a1[j]) + b2[i];



# constraints for layer 1
subject to zassign1{i in 1..l1}:
z1[i] = sum{j in 1..l0} (W1[j,i] * z0[j]) + b1[i];


