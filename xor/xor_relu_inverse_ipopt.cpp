//
//  xor_relu_inverse_ipopt.cpp
//  xor
//
//  Created by Heirich, Alan on 1/19/18.
//  Copyright © 2018 Heirich, Alan. All rights reserved.
//

#include "xor_relu_inverse_ipopt.hpp"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cstring>

// Model trained using Keras/Tensorflow with Relu units

// input weights to hidden layer
Number W1[numInputUnits][numHiddenUnits] = {
  { -0.4055348,  -0.52823716, -0.43952978, -0.18343377, -0.47216558, -0.32241216,
    -0.55611771, -0.38063344,  0.04775526, -0.76964206, -0.44537184,  0.58646101,
    0.83039683, -0.26692185, -0.48729387, -0.3081055 },
  {-0.29524794,  0.52825111, -0.21107587, -0.23073712, -0.55283833, -0.12979296,
    0.55611771, -0.36968827,  0.00462923,  0.76966035, -0.15453213, -0.58635885,
    -0.83039653, -0.04750431, -0.08500406,  0.4753032 }};
// hidden layer biases
Number b1[numHiddenUnits] =
{ 2.95232803e-01,  -1.23388522e-09,   0.00000000e+00,   0.00000000e+00,
  0.00000000e+00,   0.00000000e+00,   2.25655539e-09,   0.00000000e+00,
  2.17137873e-01,  -4.03572953e-09,   0.00000000e+00,   3.15902193e-10,
  1.22724542e-09,   0.00000000e+00,   0.00000000e+00,   3.08106124e-01 };
// hidden weights to output layer
Number W2[numHiddenUnits][numOutputUnits] = {
  { 0.3061325,   0.7249217 },
  { 0.82212615, -0.87020439},
  {-0.16003487,  0.05678433},
  { 0.03079337,  0.03488749},
  {-0.12205628, -0.03029424},
  {-0.34037149, -0.16318902},
  { 0.15318292, -1.06993532},
  {-0.17671171,  0.4517504 },
  {-0.05496955,  0.47440329},
  { 0.6847145,  -0.35256037},
  {-0.3802914,   0.24590087},
  { 0.69478261, -0.19410631},
  { 0.86003262, -0.41682884},
  {-0.32823354,  0.22413403},
  { 0.57643712, -0.20763171},
  { 0.09290985,  1.13164198}};
// output layer biases
Number b2[numOutputUnits] = {-0.10707041,  0.33430237};

/*
 * Relu constraint a=max(z,0)
 *
 * want to create a double LP constraint of the form
 * W a + b = z    if z > 0
 * W a + b <= z   if z == 0
 *
 * implement this as a set of constraints with boolean variable alpha
 * alpha * z >= 0           if z >= 0 and alpha == 1
 * alpha(W a + b - z) = 0   if z > 0 and alpha == 1
 * (1-alpha)(W a + b) <= 0  if z == 0 and alpha == 0
 * (1-alpha) z <= 0         consistent with Relu(z)=0
 * alpha(1 - alpha) = 0     ensure alpha is 0 or 1
 *
 * if the solver chooses a solution where alpha == 1 that means
 * z >= 0
 * W a + b = z
 * 0 <= 0
 * 0 <= 0
 * 0 = 0
 * which is valid no matter what, since z==0 is acceptable
 * if the solver chooses a solution where alpha == 0 that means
 * 0 >= 0
 * 0 = 0
 * W a + b <= 0
 * z <= 0
 * 0 = 0
 *
 * NLP problem: minimize f(x) s.t. gL <= g(x) <= gU and xL <= x <= xU
 * f(x) represents the distance from the target output
 *    = sum_i ( Relu(z2_i) - a2_i )^2   where a2 is the target activations
 *
 * In order for this to be differentiable we use the following definition of Relu:
 * Relu(x) = x / (1 + e^-kx)
 * d Relu(x) / d x = 2 e^2kx (kx + e^kx + 1) / (e^kx + 1)^3
 * use k=1000 if you can, this requires extended precision float and exp()
 *
 * gU can be bounded by the product of weights and activations, max activation is a recursive function back to layer 0 and max input value
 * network has an expansion/contraction ratio like an eigenvalue, tells how much the activation can grow across levels assuming weights are bounded at 1
 * use this as a network measure, relate this to the max activation that bounds gL in this case
 *
 
 * Order of unknowns:
 * Z2 (2)
 * Z1 (16)
 * Z0 (2)
 * alpha2 (2)
 * alpha1 (16)
 
 * Order of constraints:
 
 * alpha2_0 z2_0 >= 0
 * alpha2_0 (W2 z1 + b2 - z2)_0 = 0
 * (1-alpha2_0) (W2 z1 + b2)_0 <= 0
 * (1-alpha2_0) z2_0 <= 0
 * alpha2_0 (1-alpha2_0) = 0
 
 * alpha2_1 z2_1 >= 0
 * alpha2_1 (W2 a1 + b2 - z2)_1 = 0
 * (1-alpha2_1) (W2 a1 + b2)_1 <= 0
 * (1-alpha2_1) z2_1 <= 0
 * alpha2_1 (1-alpha2_1) = 0
 
 * alpha1_0 z1_0 >= 0
 * alpha1_0 (W1 z0 + b1 - z1)_0 = 0
 * (1-alpha1_0) (W1 z0 + b1)_0 <= 0
 * (1-alpha1_0) z1_0 <= 0
 * alpha1_0 (1-alpha1_0) = 0
 
 * ...
 * alpha1_i z1_i >= 0
 * alpha1_i (W1 z0 + b1 - z1)_i = 0
 * (1-alpha1_i) (W1 z0 + b1)_i <= 0
 * (1-alpha1_i) z1_i <= 0
 * alpha1_i (1-alpha1_i) = 0
 
 */


void initializeAtAFixedPoint(Number* x) {
  // case 1,0
  // initialize Z to a known solution
  Number z2_10[] = {
    9.99999865e-01,   1.19366783e-07
  };
  Number z1_10[] = {
    -1.10301997e-01,  -5.28237161e-01,  -4.39529780e-01,
    -1.83433770e-01,  -4.72165580e-01,  -3.22412160e-01,
    -5.56117708e-01,  -3.80633440e-01,   2.64893133e-01,
    -7.69642064e-01,  -4.45371840e-01,   5.86461010e-01,
    8.30396831e-01,  -2.66921850e-01,  -4.87293870e-01,
    6.24000000e-07
  };
  Number* z2 = x;
  memcpy(z2, z2_10, numOutputUnits * sizeof(Number));
  Number* z1 = z2 + numOutputUnits;
  memcpy(z1, z1_10, numHiddenUnits * sizeof(Number));
  Number* z0 = z1 + numHiddenUnits;
  z0[0] = 1;
  z0[1] = 0;
  // initialize alpha with respect to the known z
  Number* alpha2 = z0 + numInputUnits;
  for(unsigned i = 0; i < numOutputUnits; ++i) {
    alpha2[i] = (z2[i] <= 0) ? 1 : 0;
  }
  Number* alpha1 = alpha2 + numOutputUnits;
  for(unsigned i = 0; i < numHiddenUnits; ++i) {
    alpha1[i] = (z1[i] <= 0) ? 1 : 0;
  }
}


int main()
{
  Index n=-1;                          /* number of variables */
  Index m=-1;                          /* number of constraints */
  Number* x_L = NULL;                  /* lower bounds on x */
  Number* x_U = NULL;                  /* upper bounds on x */
  Number* g_L = NULL;                  /* lower bounds on g */
  Number* g_U = NULL;                  /* upper bounds on g */
  IpoptProblem nlp = NULL;             /* IpoptProblem */
  enum ApplicationReturnStatus status; /* Solve return code */
  Number* x = NULL;                    /* starting point and solution vector */
  Number* mult_g = NULL;               /* constraint multipliers
                                        at the solution */
  Number* mult_x_L = NULL;             /* lower bound multipliers
                                        at the solution */
  Number* mult_x_U = NULL;             /* upper bound multipliers
                                        at the solution */
  Number obj;                          /* objective value */
  Index i;                             /* generic counter */
  
  
  /* Number of nonzeros in the Jacobian of the constraints */
  Index nele_jac = numConstraints * numUnknowns;
  /* Number of nonzeros in the Hessian of the Lagrangian (lower or
   upper triangual part only) */
  Index nele_hess = 0;
  /* indexing style for matrices */
  Index index_style = 0; /* C-style; start counting of rows and column
                          indices at 0 */
  
  /* our user data for the function evalutions. */
  struct MyUserData user_data;
  
  /* set the number of variables and allocate space for the bounds */
  n = numUnknowns;
  x_L = (Number*)malloc(sizeof(Number)*n);
  x_U = (Number*)malloc(sizeof(Number)*n);
  
  /* set the values for the variable bounds */
  const Number unknownBound = 10;
  for (i=0; i<n; i++) {
    x_L[i] = -unknownBound;
    x_U[i] = unknownBound;
  }
  
  /* set the number of constraints and allocate space for the bounds */
  m = numConstraints;
  g_L = (Number*)malloc(sizeof(Number)*m);
  g_U = (Number*)malloc(sizeof(Number)*m);
  
  /* set the values of the constraint bounds */
  // to enforce a constraint g(x) <= 0 set g_L=-k, g_U=0
  // to enforce an equality constraint g(x) = 0 set g_L=g_U=0
  const Number lowerConstraintBound = -100;
  for(unsigned i = 0; i < m; ++i) {
    g_U[i] = 0;
    switch(i % constraintsPerAlpha) {
      case 0:
      case 2:
        g_L[i] = 0;
        break;
      case 1:
        g_L[i] = lowerConstraintBound;
        break;
      default: assert(false);
    }
  }
  
  /* create the IpoptProblem */
  nlp = CreateIpoptProblem(n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess,
                           index_style, &eval_f, &eval_g, &eval_grad_f,
                           &eval_jac_g, &eval_h);
  
  /* We can free the memory now - the values for the bounds have been
   copied internally in CreateIpoptProblem */
  free(x_L);
  free(x_U);
  free(g_L);
  free(g_U);
  
  /* Set some options. */
  //  AddIpoptStrOption(nlp, (char*)"output_file", (char*)"ipopt.out");
  AddIpoptStrOption(nlp, (char*)"hessian_approximation", (char*)"limited-memory");
  
  /* allocate space for the initial point and set the values */
  x = (Number*)calloc(n, sizeof(Number));
  initializeAtAFixedPoint(x);
  
  
  /* allocate space to store the bound multipliers at the solution */
  mult_g = (Number*)malloc(sizeof(Number)*m);
  mult_x_L = (Number*)malloc(sizeof(Number)*n);
  mult_x_U = (Number*)malloc(sizeof(Number)*n);
  
  /* Initialize the user data with the value of the network output layer. */
  // output [1,0] inputs are [1,0] or [0,1]
  user_data.ao_target[0] = 1.0;
  user_data.ao_target[1] = 0.0;
  
  /* Set the callback method for intermediate user-control.  This is
   * not required, just gives you some intermediate control in case
   * you need it. */
  /* SetIntermediateCallback(nlp, intermediate_cb); */
  
  /* solve the problem */
  status = IpoptSolve(nlp, x, NULL, &obj, mult_g, mult_x_L, mult_x_U, &user_data);
  
  if (status == Solve_Succeeded) {
    printf("\n\nSolution of the primal variables, x\n");
    for (i=0; i<n; i++) {
      printf("x[%d] = %e\n", i, x[i]);
    }
    
    printf("\n\nSolution of the ccnstraint multipliers, lambda\n");
    for (i=0; i<m; i++) {
      printf("lambda[%d] = %e\n", i, mult_g[i]);
    }
    printf("\n\nSolution of the bound multipliers, z_L and z_U\n");
    for (i=0; i<n; i++) {
      printf("z_L[%d] = %e\n", i, mult_x_L[i]);
    }
    for (i=0; i<n; i++) {
      printf("z_U[%d] = %e\n", i, mult_x_U[i]);
    }
    
    printf("\n\nObjective value\n");
    printf("f(x*) = %e\n", obj);
  }
  else {
    printf("\n\nERROR OCCURRED DURING IPOPT OPTIMIZATION.\n");
  }
  
  /* free allocated memory */
  FreeIpoptProblem(nlp);
  free(x);
  free(mult_g);
  free(mult_x_L);
  free(mult_x_U);
  
  return (int)status;
}





/* Function Implementations */

// differentiable Relu operator

const Number steepness = 10;//need extended precision for higher steepness TODO

Number Relu(Number x) {
  return x / (1 + exp(-steepness * x));
}



Number dRelu_dx(Number x) {
  return
  (2 * exp(2 * steepness * x) * (steepness * x + exp(steepness * x) + 1))
  /
  pow((exp(steepness * x) + 1), 3);
  //TODO use extended precision library and exp fuction, cast result to Number
  
  // derivative by WolframAlpha
  //(2 x e^(2 k x) (k x + e^(k x) + 1))/(e^(k x) + 1)^3
}


void affine1(Number W1[numInputUnits][numHiddenUnits],
             Number z0[numInputUnits],
             Number b1[numHiddenUnits],
             Number z1[numHiddenUnits])
{
  // TODO use BLAS
  for(unsigned i = 0; i < numHiddenUnits; ++i) {
    z1[i] = 0;
    for(unsigned j = 0; j < numInputUnits; ++j) {
      z1[i] += W1[j][i] * z0[j];
    }
    z1[i] += b1[i];
  }
}



void affine2(Number W2[numHiddenUnits][numOutputUnits],
             Number z1[numHiddenUnits],
             Number b2[numOutputUnits],
             Number z2[numOutputUnits])
{
  // TODO use BLAS
  for(unsigned i = 0; i < numOutputUnits; ++i) {
    z2[i] = 0;
    for(unsigned j = 0; j < numHiddenUnits; ++j) {
      z2[i] += W2[j][i] * z1[j];
    }
    z2[i] += b2[i];
  }
}

void activation2(Number W2[numHiddenUnits][numOutputUnits],
                 Number z1[numHiddenUnits],
                 Number b2[numOutputUnits],
                 Number a2[numOutputUnits])
{
  Number z2[numOutputUnits];
  for(unsigned i = 0; i < numOutputUnits; ++i) {
    z2[i] = 0;
    for(unsigned j = 0; j < numHiddenUnits; ++j) {
      z2[i] += W2[j][i] * Relu(z1[j]);
    }
    z2[i] += b2[i];
  }
  for(unsigned i = 0; i < numOutputUnits; ++i) {
    a2[i] = Relu(z2[i]);
  }
}


void printUnknowns(Number* x) {
  static unsigned iteration = 0;
  Number* z2 = x;
  Number* z1 = z2 + numOutputUnits;
  Number* z0 = z1 + numHiddenUnits;
  Number* alpha2 = z0 + numInputUnits;
  Number* alpha1 = alpha2 + numOutputUnits;
  std::cout << "=== iteration " << iteration++ << "===" << std::endl;
  
  std::cout << "z2: ";
  for(unsigned i = 0; i < numOutputUnits; ++i) {
    std::cout << z2[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "z1: ";
  for(unsigned i = 0; i < numHiddenUnits; ++i) {
    std::cout << z1[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "z0: ";
  for(unsigned i = 0; i < numInputUnits; ++i) {
    std::cout << z0[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "alpha2: ";
  for(unsigned i = 0; i < numOutputUnits; ++i) {
    std::cout << alpha2[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "alpha1: ";
  for(unsigned i = 0; i < numHiddenUnits; ++i) {
    std::cout << alpha1[i] << " ";
  }
  std::cout << std::endl;
}


// f is the sum squared error at the output layer

Bool eval_f(Index n, Number* x, Bool new_x,
            Number* obj_value, UserDataPtr user_data)
{
  assert(n == numUnknowns);
  printUnknowns(x);
  Number* z1 = x + numInputUnits;
  Number a_o_computed[numOutputUnits];
  activation2(W2, z1, b2, a_o_computed);
  struct MyUserData* data = (struct MyUserData*)user_data;
  Number* a_o_target = data->ao_target;
  Number sumSquaredError = 0;
  for(unsigned i = 0; i < numOutputUnits; ++i) {
    Number error = a_o_computed[i] - a_o_target[i];
    sumSquaredError += error * error;
  }
  *obj_value = sumSquaredError;
  //
  std::cout << "ao_computed: " << a_o_computed[0] << " " << a_o_computed[1] << std::endl;
  std::cout << "objective: " << sumSquaredError << std::endl;
  //
  return TRUE;
}


// gradient of f by WolframAlpha
/* k - steepness
 * a - a_o[0] ground truth
 * b - a_o[1]
 * x - z_computed[0]
 * y - z_computed[1]
 *
 * grad((x/(e^(-k x) + 1) - a)^2 + (y/(e^(-k y) + 1) - b)^2) =
 * (-(2 e^(k x) (k x + e^(k x) + 1) (a e^(k x) + a - x e^(k x)))/(e^(k x) + 1)^3,
 * -(2 e^(k y) (k y + e^(k y) + 1) (b e^(k y) + b - y e^(k y)))/(e^(k y) + 1)^3)
 *
 * f = sum_i ( Relu(W2.z1+b2)_i - ao_i )^2
 *
 * grad_f = 2 e^2k k ( k + e^k + 1 ) / ( e^k + 1 )^3
 *
 */
//TODO use extended precision library and exp fuction, cast result to Number


Bool eval_grad_f(Index n, Number* x, Bool new_x,
                 Number* grad_f, UserDataPtr user_data)
{
  assert(n == numUnknowns);
  Number* z1 = x + numInputUnits;
  Number z2_computed[numOutputUnits];
  affine2(W2, z1, b2, z2_computed);
  MyUserData* myUserData = (MyUserData*)user_data;
  Number* a_o = myUserData->ao_target;
  std::cout << "grad f: " << std::endl;
  for(unsigned i = 0; i < numOutputUnits; ++i) {
    Number x = z2_computed[i];
    Number a = a_o[i];
    Number k = steepness;
    Number numerator = -(2 * exp(k * x) * (k * x + exp(k * x) + 1) * (a * exp(k * x) + a - x * exp(k * x)));
    Number denominator = pow(exp(k * x) + 1, 3);
    grad_f[i] = numerator /denominator;
    std::cout << "x " << x << " a " << a << " k " << k << std::endl;
    std::cout << "grad_f[" << i << "] = " << grad_f[i] << " = " << numerator << " / " << denominator << std::endl;
  }
  std::cout << std::endl;
  return TRUE;
}




Bool eval_g(Index n, Number* x, Bool new_x,
            Index m, Number* g, UserDataPtr user_data)
{
  assert(n == numUnknowns);
  assert(m == numConstraints);
  
  Number* z2 = x;
  Number* z1 = x + numOutputUnits;
  Number* z0 = z1 + numHiddenUnits;
  Number* alpha2 = x + numActivations;
  Number* alpha1 = alpha2 + numOutputUnits;
  Number* gNext = g;
  Number* gPtr = g;

  Number z2_computed[numOutputUnits];
  affine2(W2, z1, b2, z2_computed);
  std::cout << std::endl << "constraints for z2:" << std::endl;
  unsigned gIdx = 0;
  
  for(unsigned i = 0; i < numOutputUnits; ++i) {
    std::cout << "alpha2_" << i << " = " << alpha2[i] << std::endl;
    
#if FIVE_CONSTRAINTS
    *gNext++ = alpha2[i] * z2[i] * -1;
    if(alpha2[i] > 0.5) {
      std::cout << "g[" << gIdx++ << "] " << alpha2[i] << " * z2_" << i << " " << z2[i] << " * -1 = " << *gPtr++ << " <=? 0" << std::endl;
    } else {
      std::cout << "g[" << gIdx++ << "] " << alpha2[i] << " * " << z2[i] << " = " << *gPtr++ << " <=? 0" << std::endl;
    }
#endif
    
    *gNext++ = alpha2[i] * (z2_computed[i] - z2[i]);
    if(alpha2[i] > 0.5) {
      std::cout << "g[" << gIdx++ << "] " << alpha2[i] << " * (W2.z1+b2-z2)_" << i << " " << (z2_computed[i] - z2[i]) << " = " << *gPtr++ << " =? 0" << std::endl;
    } else {
      std::cout << "g[" << gIdx++ << "] " << alpha2[i] << " * " << (z2_computed[i] - z2[i]) << " = " << *gPtr++ << " =? 0" << std::endl;
    }
    
    *gNext++ = (1 - alpha2[i]) * z2_computed[i];
    if(alpha2[i] > 0.5) {
      std::cout << "g[" << gIdx++ << "] " << (1 - alpha2[i]) << " * " << z2_computed[i] << " = " << *gPtr++ << " <=? 0" << std::endl;
    } else {
      std::cout << "g[" << gIdx++ << "] " << (1 - alpha2[i]) << " * (W2.z1+b2) " << z2_computed[i] << " = " << *gPtr++ << " <=? 0" << std::endl;
    }
    
#if FIVE_CONSTRAINTS
    *gNext++ = (1 - alpha2[i]) * z2[i];
    if(alpha2[i] > 0.5) {
      std::cout << "g[" << gIdx++ << "] " << (1 - alpha2[i]) << " * " << z2[i] << " = " << *gPtr++ << " <=? 0" << std::endl;
    } else {
      std::cout << "g[" << gIdx++ << "] " << (1 - alpha2[i]) << " * z2_" << i << " " << z2[i] << " = " << *gPtr++ << " <=? 0" << std::endl;
    }
#endif
    
    *gNext++ = alpha2[i] * (1 - alpha2[i]);
    std::cout << "g[" << gIdx++ << "] " << alpha2[i] << " * (1 - " << alpha2[i] << ") = " << alpha2[i] * (1 - alpha2[i]) << " = " << *gPtr++ << " =? 0" << std::endl;
  }
  std::cout << std::endl;
  
  
  
  Number z1_computed[numHiddenUnits];
  affine1(W1, z0, b1, z1_computed);
  std::cout << std::endl << "constraints for z1:" << std::endl;
  
  for(unsigned i = 0; i < numHiddenUnits; ++i) {
    std::cout << "alpha1_" << i << " = " << alpha1[i] << std::endl;
    
#if FIVE_CONSTRAINTS
    *gNext++ = alpha1[i] * z1[i] * -1;
    if(alpha1[i] > 0.5) {
      std::cout << "g[" << gIdx++ << "] " << alpha1[i] << " * z1_" << i << " " << z1[i] << " * -1 = " << *gPtr++ << " <=? 0" << std::endl;
    } else {
      std::cout << "g[" << gIdx++ << "] " << alpha1[i] << " * " << z1[i] << " = " << *gPtr++ << " <=? 0" << std::endl;
    }
#endif
    
    *gNext++ = alpha1[i] * (z1_computed[i] - z1[i]);
    if(alpha1[i] > 0.5) {
      std::cout << "g[" << gIdx++ << "] " << alpha1[i] << " * (W1.z0+b1-z1)_" << i << " " << (z1_computed[i] - z1[i]) << " = " << *gPtr++ << " =? 0" << std::endl;
    } else {
      std::cout << "g[" << gIdx++ << "] " << alpha1[i] << " * " << (z1_computed[i] - z1[i]) << " = " << *gPtr++ << " =? 0" << std::endl;
    }
    
    *gNext++ = (1 - alpha1[i]) * z1_computed[i];
    if(alpha1[i] > 0.5) {
      std::cout << "g[" << gIdx++ << "] " << (1 - alpha1[i]) << " * " << z1_computed[i] << " = " << *gPtr++ << " <=? 0" << std::endl;
    } else {
      std::cout << "g[" << gIdx++ << "] " << (1 - alpha1[i]) << " * (W1.z0+b1) " << z1_computed[i] << " = " << *gPtr++ << " <=? 0" << std::endl;
    }
    
#if FIVE_CONSTRAINTS
    *gNext++ = (1 - alpha1[i]) * z1[i];
    if(alpha1[i] > 0.5) {
      std::cout << "g[" << gIdx++ << "] " << (1 - alpha1[i]) << " * " << z1[i] << " = " << *gPtr++ << " <=? 0" << std::endl;
    } else {
      std::cout << "g[" << gIdx++ << "] " << (1 - alpha1[i]) << " * z1_" << i << " " << z1[i] << " = " << *gPtr++ << " <=? 0" << std::endl;
    }
#endif
    
    *gNext++ = alpha1[i] * (1 - alpha1[i]);
    std::cout << "g[" << gIdx++ << "] " << alpha1[i] << " * (1 - " << alpha1[i] << ") = " << alpha1[i] * (1 - alpha1[i]) << " = " << *gPtr++ << " =? 0" << std::endl;
  }
  std::cout << std::endl;
  
  
  return TRUE;
}



void generateConstraintJacobian(Number* values, unsigned& numPoints, bool layer2, unsigned numUnits, Number* z, Number* z_computed, Number* alpha) {
  
  for(unsigned unit = 0; unit < numUnits; ++unit) {
    
#if FIVE_CONSTRAINTS

    ///// g[0] /////
    
    // d g[0] / d z2
    
    for(unsigned i = 0; i < numOutputUnits; i++) {
      if(i == unit && layer2) {
        values[numPoints++] = -alpha[unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[0] / d z1
    
    for(unsigned i = 0; i < numHiddenUnits; i++) {
      if(i == unit && !layer2) {
        values[numPoints++] = -alpha[unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[0] / d z0
    
    for(unsigned i = 0; i < numInputUnits; i++) {
      values[numPoints++] = 0;
    }
    
    // d g[0] / d alpha2
    
    for(unsigned i = 0; i < numOutputUnits; i++) {
      if(i == unit && layer2) {
        values[numPoints++] = -z[unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[0] / d alpha1
    
    for(unsigned i = 0; i < numHiddenUnits; i++) {
      if(i == unit && !layer2) {
        values[numPoints++] = -z[unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    assert(numPoints % numUnknowns == 0);
    
#endif

    
    ///// g[1] /////
    
    // d g[1] / d z2
    
    for(unsigned i = 0; i < numOutputUnits; i++) {
      if(i == unit && layer2) {
        values[numPoints++] = alpha[unit] * -1;
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[1] / d z1
    
    for(unsigned j = 0; j < numHiddenUnits; ++j) {
      if(layer2) {
        values[numPoints++] = alpha[unit] * W2[j][unit];
      } else {
        if(j == unit) {
          values[numPoints++] = -1 * alpha[unit];
        } else {
          values[numPoints++] = 0;
        }
      }
    }
    
    // d g[1] / d z0
    
    for(unsigned j = 0; j < numInputUnits; ++j) {
      if(!layer2) {
        values[numPoints++] = alpha[unit] * W1[j][unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[1] / d alpha2
    
    for(unsigned i = 0; i < numOutputUnits; i++) {
      if(i == unit && layer2) {
        values[numPoints++] = z_computed[unit] - z[unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[1] / d alpha1
    
    for(unsigned i = 0; i < numHiddenUnits; ++i) {
      if(i == unit && !layer2) {
        values[numPoints++] = z_computed[unit] - z[unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    assert(numPoints % numUnknowns == 0);

    
    ///// g[2] /////
    
    // d g[2] / d z2
    
    for(unsigned i = 0; i < numOutputUnits; i++) {
      values[numPoints++] = 0;
    }
    
    // d g[2] / d z1
    
    for(unsigned i = 0; i < numHiddenUnits; ++i) {
      if(layer2) {
        values[numPoints++] = (1 - alpha[unit]) * W2[i][unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[2] / d z0
    
    for(unsigned i = 0; i < numInputUnits; ++i) {
      if(!layer2) {
        values[numPoints++] = (1 - alpha[unit]) * W1[i][unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[2] / d alpha2
    
    for(unsigned i = 0; i < numOutputUnits; i++) {
      if(i == unit && layer2) {
        values[numPoints++] = -1 * z_computed[unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[2] / d alpha1
    
    for(unsigned i = 0; i < numHiddenUnits; ++i) {
      if(i == unit && !layer2) {
        values[numPoints++] = -1 * z_computed[unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    assert(numPoints % numUnknowns == 0);
    
#if FIVE_CONSTRAINTS

    ///// g[3] /////
    
    // d g[3] / d z2
    
    for(unsigned i = 0; i < numOutputUnits; i++) {
      if(i == unit && layer2) {
        values[numPoints++] = (1 - alpha[unit]);
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[3] / d z1
    
    for(unsigned i = 0; i < numHiddenUnits; ++i) {
      if(i == unit && !layer2) {
        values[numPoints++] = (1 - alpha[unit]);
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[3] / d z0
    
    for(unsigned i = 0; i < numInputUnits; ++i) {
      values[numPoints++] = 0;
    }
    
    // d g[3] / d alpha2
    
    for(unsigned i = 0; i < numOutputUnits; i++) {
      if(i == unit && layer2) {
        values[numPoints++] = -z[unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[3] / d alpha1
    
    for(unsigned i = 0; i < numHiddenUnits; ++i) {
      if(i == unit && !layer2) {
        values[numPoints++] = -z[unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    assert(numPoints % numUnknowns == 0);
    
#endif
    
    ///// g[4] /////
    
    // d g[4] / d z
    
    for(unsigned i = 0; i < numActivations; ++i) {
      values[numPoints++] = 0;
    }
    
    // d g[4] / d alpha2
    
    for(unsigned i = 0; i < numOutputUnits; i++) {
      if(i == unit && layer2) {
        values[numPoints++] = 1 - 2 * alpha[unit];
      } else {
        values[numPoints++] = 0;
      }
    }
    
    // d g[4] / d alpha1
    
    for(unsigned i = 0; i < numHiddenUnits; ++i) {
      if(i == unit && !layer2) {
        values[numPoints++] = 1 - 2 * alpha[unit];
      } else {
        values[numPoints++] = 0;
      }
    }
  }
}




Bool eval_jac_g(Index n, Number *x, Bool new_x,
                Index m, Index nele_jac,
                Index *iRow, Index *jCol, Number *values,
                UserDataPtr user_data)
{
  assert(n == numUnknowns);
  assert(m == numConstraints);
  
  
  
  if (values == NULL) {
    /* return the structure of the jacobian */
    
    /* this particular jacobian is dense */
    unsigned numPoints = 0;
    
    for(unsigned i = 0; i < numConstraints; ++i) {
      for(unsigned j = 0; j < numUnknowns; ++j) {
        iRow[numPoints] = i;
        jCol[numPoints] = j;
        numPoints++;
      }
    }
    
  }
  else {
    /* return the values of the jacobian of the constraints */
    
    Number* z2 = x;
    Number* z1 = x + numOutputUnits;
    Number* z0 = z1 + numHiddenUnits;
    Number* alpha2 = x + numActivations;
    Number* alpha1 = alpha2 + numOutputUnits;
    
    Number z2_computed[numOutputUnits];
    affine2(W2, z1, b2, z2_computed);
    Number z1_computed[numHiddenUnits];
    affine1(W1, z0, b1, z1_computed);
    
    unsigned numPoints = 0;
    generateConstraintJacobian(values, numPoints, true, numOutputUnits, z2, z2_computed, alpha2);
    generateConstraintJacobian(values, numPoints, false, numHiddenUnits, z1, z1_computed, alpha1);
    
    assert(numPoints == numUnknowns *  numConstraints);
    
    std::cout << "Jacobian of constraint grad_g:" << std::endl;
    unsigned valuesIdx = 0;
    for(unsigned gIdx = 0; gIdx < numConstraints; ++gIdx) {
      if(gIdx % constraintsPerAlpha == 0) {
        unsigned unit = gIdx / constraintsPerAlpha;
        if (unit < numOutputUnits) {
          unsigned u = unit;
          std::cout << "unit " << u << " layer 2" << std::endl;
        } else if (unit < numOutputUnits + numHiddenUnits) {
          std::cout<<unit<<std::endl;
          unsigned u = unit - numOutputUnits;
          std::cout << "unit " << u << " layer 1" << std::endl;
        }
      }
      std::cout << gIdx << ": ";
      for(unsigned i = 0; i < numUnknowns; ++i) {
        if(i == numOutputUnits
           || i == numOutputUnits + numHiddenUnits
           || i == numActivations
           || i == numActivations + numOutputUnits
           || i == numActivations + numOutputUnits + numHiddenUnits) {
          std::cout << "\t| ";
        }
        std::cout << values[valuesIdx++] << " ";
      }
      std::cout << std::endl;
    }
  }
  
  return TRUE;
}





Bool eval_h(Index n, Number *x, Bool new_x, Number obj_factor,
            Index m, Number *lambda, Bool new_lambda,
            Index nele_hess, Index *iRow, Index *jCol,
            Number *values, UserDataPtr user_data)
{
  // we are using the quasi-newton approximation of second derivatives
  // so we don't need the hessian of the lagrangian
  return FALSE;
}



Bool intermediate_cb(Index alg_mod, Index iter_count, Number obj_value,
                     Number inf_pr, Number inf_du, Number mu, Number d_norm,
                     Number regularization_size, Number alpha_du,
                     Number alpha_pr, Index ls_trials, UserDataPtr user_data)
{
  printf("Testing intermediate callback in iteration %d\n", iter_count);
  if (inf_pr < 1e-4) return FALSE;
  return TRUE;
}


