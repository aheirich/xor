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


// input weights to hidden layer
Number W1[numInputUnits][numHiddenUnits] = {
  { 0.9145419,  -0.5964824,  -0.642739,   -0.34978598,  0.41890582,  0.22138357,
    -0.08724618, -0.17340243, -0.06329376,  0.45263577},
  {-0.88043803,  0.70016438,  0.94026381,  0.19096281, -0.66918135,  0.26964825,
    -0.33756,    -0.1920374,  -0.11555147,  0.92417163 }};
// hidden layer biases
Number b1[numHiddenUnits] =
{ -1.29862165e-08,  -2.89841706e-10,  -3.04766568e-09,  -1.96804002e-01,
  -3.54976351e-08,   6.22820675e-01,   0.00000000e+00,   0.00000000e+00,
  0.00000000e+00,  -4.52658087e-01 };
// hidden weights to output layer
Number W2[numHiddenUnits][numOutputUnits] = {
  { 0.99346685, -0.81308043},
  { 0.60336119, -0.51245868},
  { 0.79297972, -0.97627968},
  {-0.19056943, -0.17135906},
  { 0.17632513, -0.93301785},
  { 0.07936023,  0.60728908},
  { 0.29867882, -0.16222471},
  { 0.3594684,   0.53735191},
  {-0.39656982, -0.59148967},
  {-0.4018164,   0.07913463 }};
// output layer biases
Number b2[numOutputUnits] = {-0.049427,    0.62176669};

/*
 * Relu constraint a=max(z,0)
 *
 * want to create a double LP constraint of the form
 * W a + b = z    if z > 0
 * W a + b <= z   if z == 0
 *
 * implement this as a set of constraints with boolean variable alpha
 * alpha(W a + b) = z       if z > 0 and alpha == 1
 * (1-alpha)(W a + b) <= z  if z == 0 and alpha == 0
 * alpha(1 - alpha) = 0     ensure alpha is 0 or 1
 *
 * if the solver chooses a solution where alpha == 1 that means
 * W a + b = z
 * 0 <= z
 * which is valid no matter what, since z==0 is acceptable
 * if the solver chooses a solution where alpha == 0 that means
 * 0 = z
 * W a + b <= z
 *
 * NLP problem: minimize f(x) s.t. gL <= g(x) <= gU and xL <= x <= xU
 *
 * f(x) = sum_i (a_o,i - a'_o,i)^2 where a' is the given target activation, a is computed from the unknowns
 *
 * alpha (W a + b) = alpha(z)
 * alpha_l (W_l . a_l + b_l - z_l) = 0
 * gL = gU = 0
 *
 * (1-alpha_l) (W_l . a_l + b_l - z_l) <= 0
 * gU = 0
 * gL can be bounded by the product of weights and activations, max activation is a recursive function back to layer 0 and max input value
 * network has an expansion/contraction ratio like an eigenvalue, tells how much the activation can grow across levels assuming weights are bounded at 1
 * propose this as a network measure, relate this to the max activation that bounds gL in this case
 *
 * alpha_l (1 - alpha_l) = 0
 * gL = gU = 0
 *
 
 * Order of unknowns:
 * Z2 (2)
 * Z1 (10)
 * a0 (2)
 * alpha2 (2)
 * alpha1 (10)
 
 * Order of constraints:
 
 * alpha2_0 (W2 z1 + b2 - z2)_0 = 0
 * (1 - alpha2_0) (W2 z1 + b2 - z2)_0 <= 0
 * alpha2_0 ( 1 - alpha2_0 ) = 0
 
 * alpha2_1 (W2 a1 + b2 - z2)_1 = 0
 * (1 - alpha2_1) (W2 a1 + b2 - z2)_1 <= 0
 * alpha2_1 ( 1 - alpha2_1 ) = 0
 
 * alpha1_0 (W1 z0 + b1 - z1)_0 = 0
 * (1 - alpha1_0) (W1 z0 + b1 -z1)_0 <= 0
 * alpha1_0 ( 1 - alpha1_0 ) = 0
 
 * ...
 * alpha1_i (W1 z0 + b1 - z1)_i = 0
 * (1 - alpha1_i) (W1 z0 + b1 -z1)_i <= 0
 * alpha1_i (1 - alpha1_i) = 0
 
 * ...
 * z0_0 >= 0
 * z1_0 >= 0
 
 */



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
  for (i=0; i<n; i++) {
    x_L[i] = -5.0;
    x_U[i] = 5.0;
  }
  
  /* set the number of constraints and allocate space for the bounds */
  m = numConstraints;
  g_L = (Number*)malloc(sizeof(Number)*m);
  g_U = (Number*)malloc(sizeof(Number)*m);
  /* set the values of the constraint bounds */
  for(unsigned i = 0; i < m; ++i) {
    g_U[i] = 0;
    if(i % 3 == 1) {
      g_L[i] = -10.0;
    } else {
      g_L[i] = 0;
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
  
  /* Set some options.  Note the following ones are only examples,
   they might not be suitable for your problem. */
  AddIpoptNumOption(nlp, (char*)"tol", 1e-7);
  AddIpoptStrOption(nlp, (char*)"mu_strategy", (char*)"adaptive");
  AddIpoptStrOption(nlp, (char*)"output_file", (char*)"ipopt.out");
  
  /* allocate space for the initial point and set the values */
  x = (Number*)malloc(sizeof(Number)*n);
  x[0] = 1.0;
  x[1] = 5.0;
  x[2] = 5.0;
  x[3] = 1.0;
  
  /* allocate space to store the bound multipliers at the solution */
  mult_g = (Number*)malloc(sizeof(Number)*m);
  mult_x_L = (Number*)malloc(sizeof(Number)*n);
  mult_x_U = (Number*)malloc(sizeof(Number)*n);
  
  /* Initialize the user data */
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
  
  /* Now we are going to solve this problem again, but with slightly
   modified constraints.  We change the constraint offset of the
   first constraint a bit, and resolve the problem using the warm
   start option. */
  user_data.g_offset[0] = 0.2;
  
  if (status == Solve_Succeeded) {
    /* Now resolve with a warmstart. */
    AddIpoptStrOption(nlp, (char*)"warm_start_init_point", (char*)"yes");
    /* The following option reduce the automatic modification of the
     starting point done my Ipopt. */
    AddIpoptNumOption(nlp, (char*)"bound_push", 1e-5);
    AddIpoptNumOption(nlp, (char*)"bound_frac", 1e-5);
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
      printf("\n\nERROR OCCURRED DURING IPOPT OPTIMIZATION WITH WARM START.\n");
    }
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

Number Relu(Number x) {
  return fmax(x, 0);//can we make this differentiable using Dirac delta?
}


void affine1(Number W1[numInputUnits][numHiddenUnits],
             Number z0[numInputUnits],
             Number b1[numHiddenUnits],
             Number z1[numHiddenUnits])
{
  for(unsigned i = 0; i < numHiddenUnits; ++i) {
    z1[i] = 0;
    for(unsigned j = 0; j < numInputUnits; ++j) {
      z1[i] += W1[j][i] * z0[j];
    }
    z1[i] += b1[i];
  }
}

void activation1(Number W1[numInputUnits][numHiddenUnits],
                 Number z0[numInputUnits],
                 Number b1[numHiddenUnits],
                 Number a1[numHiddenUnits])
{
  Number z1[numHiddenUnits];
  affine1(W1, z0, b1, z1);
  for(unsigned i = 0; i < numHiddenUnits; ++i) {
    a1[i] = Relu(z1[i]);
  }
}


void affine2(Number W2[numHiddenUnits][numOutputUnits],
             Number z1[numHiddenUnits],
             Number b2[numOutputUnits],
             Number z2[numOutputUnits])
{
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
  affine2(W2, z1, b2, z2);
  for(unsigned i = 0; i < numOutputUnits; ++i) {
    a2[i] = Relu(z2[i]);
  }
}


// f is the sum squared error at the output layer

Bool eval_f(Index n, Number* x, Bool new_x,
            Number* obj_value, UserDataPtr user_data)
{
  assert(n == numUnknowns);
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
  return TRUE;
}


// gradient of f

Bool eval_grad_f(Index n, Number* x, Bool new_x,
                 Number* grad_f, UserDataPtr user_data)
{
  assert(n == numUnknowns);
  struct MyUserData* data = (struct MyUserData*)user_data;
  Number* a_o_target = data->ao_target;
  Number* z_o = x;
  for(unsigned i = 0; i < numOutputUnits; ++i) {
    grad_f[i] = 2 * (Relu(z_o[i]) - a_o_target[i]);
  }
  return TRUE;
}


/* constraints g
 * alpha_l (W_l . a_l + b_l - z_l) = 0
 * (1-alpha_l) (W_l . a_l + b_l - z_l) <= 0
 * alpha_l (1 - alpha_l) = 0
 */


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
  
  Number z2_computed[numOutputUnits];
  affine2(W2, z1, b2, z2_computed);
  for(unsigned i = 0; i < numOutputUnits; ++i) {
    *gNext++ = alpha2[i] * (z2_computed[i] - z2[i]); // == z if z > 0
    *gNext++ = (1 - alpha2[i]) * z2_computed[i]; // <= z if z == 0
    *gNext++ = alpha2[i] * (1 - alpha2[i]);
  }
  
  Number z1_computed[numHiddenUnits];
  affine1(W1, z0, b1, z1_computed);
  for(unsigned i = 0; i < numHiddenUnits; ++i) {
    *gNext++ = alpha1[i] * (z1_computed[i] - z1[i]); // == z if z > 0
    *gNext++ = (1 - alpha1[i]) * z1_computed[i]; // <= z if z == 0
    *gNext++ = alpha1[i] * (1 - alpha1[i]);
  }
  return TRUE;
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
    
    unsigned numPoints = 0;
    
    // constraints for output units
    
    Number z2_computed[numOutputUnits];
    affine2(W2, z1, b2, z2_computed);
    
    
    
    ///// g[0] /////
    
    // d g[0] / d z2
    
    values[numPoints++] = alpha2[0] * -1;
    values[numPoints++] = 0;
    
    // d g[0] / d z1
    
    for(unsigned j = 0; j < numHiddenUnits; ++j) {
      values[numPoints++] = alpha2[0] * W2[j][0];
    }
    
    // d g[0] / d z0
    
    for(unsigned i = 0; i < numInputUnits; ++i) {
      values[numPoints++] = 0;
    }
    
    // d g[0] / d alpha2
    
    values[numPoints++] = z2_computed[0] - z2[0];
    values[numPoints++] = 0;
    
    // d g[0] / d alpha1
    
    for(unsigned i = 0; i < numHiddenUnits; ++i) {
      values[numPoints++] = 0;
    }
      
    
    
    ///// g[1] /////
    
    // d g[1] / d z2
    
    values[numPoints++] = 0;
    values[numPoints++] = 0;
    
    // d g[1] / d z1
    
    for(unsigned j = 0; j < numHiddenUnits; ++j) {
      values[numPoints++] = alpha2[0] * W2[j][0];
    }
    
    // d g[1] / d z0
    
    for(unsigned i = 0; i < numInputUnits; ++i) {
      values[numPoints++] = 0;
    }
    
    // d g[1] / d alpha2
    
    values[numPoints++] = z2_computed[0];
    values[numPoints++] = 0;
    
    // d g[1] / d alpha1
    
    for(unsigned i = 0; i < numHiddenUnits; ++i) {
      values[numPoints++] = 0;
    }
    
    
    
    ///// g[2] /////
    
    // d g[2]  d z*
    
    for(unsigned i = 0; i < numActivations; ++i) {
      values[numPoints++] = 0;
    }
    
    // d g[2] / d alpha2
    
    values[numPoints++] = 1 - 2 * alpha2[0];
    values[numPoints++] = 0;
    
    // d g[2] / d alpha1
    
    for(unsigned i = 0; i < numHiddenUnits; ++i) {
      values[numPoints++] = 0;
    }
    
    
    
    
    ///// g[3] /////
    
    // d g[3] / d z2
    
    values[numPoints++] = 0;
    values[numPoints++] = alpha2[1] * -1;
    
    // d g[3] / d z1
    
    for(unsigned j = 0; j < numHiddenUnits; ++j) {
      values[numPoints++] = alpha2[1] * W2[j][1];
    }
    
    // d g[3] / d z0
    
    for(unsigned i = 0; i < numInputUnits; ++i) {
      values[numPoints++] = 0;
    }
    
    // d g[3] / d alpha2
    
    values[numPoints++] = 0;
    values[numPoints++] = z2_computed[1] - z2[1];
    
    // d g[3] / d alpha1
    
    for(unsigned i = 0; i < numHiddenUnits; ++i) {
      values[numPoints++] = 0;
    }
    
    
    
    
    ///// g[4] /////
    
    // d g[4] / d z2
    
    values[numPoints++] = 0;
    values[numPoints++] = 0;
    
    // d g[4] / d z1
    
    for(unsigned j = 0; j < numHiddenUnits; ++j) {
      values[numPoints++] = alpha2[1] * W2[j][1];
    }
    
    // d g[4] / d z0
    
    for(unsigned i = 0; i < numInputUnits; ++i) {
      values[numPoints++] = 0;
    }
    
    // d g[4] / d alpha2
    
    values[numPoints++] = 0;
    values[numPoints++] = z2_computed[1];
    
    // d g[4] / d alpha1
    
    for(unsigned i = 0; i < numHiddenUnits; ++i) {
      values[numPoints++] = 0;
    }
    
    
    ///// g[5] /////
    
    // d g[5]  d z*
    
    for(unsigned i = 0; i < numActivations; ++i) {
      values[numPoints++] = 0;
    }
    
    // d g[5] / d alpha2
    
    values[numPoints++] = 0;
    values[numPoints++] = 1 - 2 * alpha2[1];
    
    // d g[5] / d alpha1
    
    for(unsigned i = 0; i < numHiddenUnits; ++i) {
      values[numPoints++] = 0;
    }
    
    
    ///// g[6..35] /////
    
    Number z1_computed[numHiddenUnits];
    affine1(W1, z0, b1, z1_computed);
    
    for(unsigned unit = 0; unit < numHiddenUnits; ++unit) {
      
      // d g[6] / d z2
 
      values[numPoints++] = 0;
      values[numPoints++] = 0;

      // d g[6] / d z1
      
      for(unsigned i = 0; i < numHiddenUnits; ++i) {
        if(i == unit) {
          values[numPoints++] = alpha1[unit] * -1;
        } else {
          values[numPoints++] = 0;
        }
      }
      
      // d g[6] / d z0
      
      values[numPoints++] = alpha1[unit] * W1[0][unit];
      values[numPoints++] = alpha1[unit] * W1[1][unit];

      // d g[6] / d alpha2
      
      values[numPoints++] = 0;
      values[numPoints++] = 0;
      
      // d g[6] / d alpha1
      
      for(unsigned i = 0; i < numHiddenUnits; ++i) {
        if(i == unit) {
          values[numPoints++] = z1_computed[unit] - z1[unit];
        } else {
          values[numPoints++] = 0;
        }
      }
      
      
      // d g[7] / d z2
      
      values[numPoints++] = 0;
      values[numPoints++] = 0;

      // d g[7] / d z1
      
      for(unsigned i = 0; i < numHiddenUnits; ++i) {
        values[numPoints++] = 0;
      }
      
      // d g[7] / d z0
      
      values[numPoints++] = -1 * alpha1[unit] * W1[0][unit];
      values[numPoints++] = -1 * alpha1[unit] * W1[1][unit];

      // d g[7] / d alpha2
      
      values[numPoints++] = 0;
      values[numPoints++] = 0;
      
      // d g[7] / d alpha1

      for(unsigned i = 0; i < numHiddenUnits; ++i) {
        if(i == unit) {
          values[numPoints++] = z1_computed[unit];
        } else {
          values[numPoints++] = 0;
        }
      }

      
      // d g[8] / d z, d alpha2
      
      for(unsigned i = 0; i < numActivations + numOutputUnits; ++i) {
        values[numPoints++] = 0;
      }
      
      // d g[8] / d alpha1
      
      for(unsigned i = 0; i < numHiddenUnits; ++i) {
        if(i == unit) {
          values[numPoints++] = 1 - 2 * alpha1[unit];
        } else {
          values[numPoints++] = 0;
        }
      }

    }


  }
  
  return TRUE;
}




Bool eval_h(Index n, Number *x, Bool new_x, Number obj_factor,
            Index m, Number *lambda, Bool new_lambda,
            Index nele_hess, Index *iRow, Index *jCol,
            Number *values, UserDataPtr user_data)
{
  Index idx = 0; /* nonzero element counter */
  Index row = 0; /* row counter for loop */
  Index col = 0; /* col counter for loop */
  if (values == NULL) {
    /* return the structure. This is a symmetric matrix, fill the lower left
     * triangle only. */
    
    /* the hessian for this problem is actually dense */
    idx=0;
    for (row = 0; row < 4; row++) {
      for (col = 0; col <= row; col++) {
        iRow[idx] = row;
        jCol[idx] = col;
        idx++;
      }
    }
    
    assert(idx == nele_hess);
  }
  else {
    /* return the values. This is a symmetric matrix, fill the lower left
     * triangle only */
    
    /* fill the objective portion */
    values[0] = obj_factor * (2*x[3]);               /* 0,0 */
    
    values[1] = obj_factor * (x[3]);                 /* 1,0 */
    values[2] = 0;                                   /* 1,1 */
    
    values[3] = obj_factor * (x[3]);                 /* 2,0 */
    values[4] = 0;                                   /* 2,1 */
    values[5] = 0;                                   /* 2,2 */
    
    values[6] = obj_factor * (2*x[0] + x[1] + x[2]); /* 3,0 */
    values[7] = obj_factor * (x[0]);                 /* 3,1 */
    values[8] = obj_factor * (x[0]);                 /* 3,2 */
    values[9] = 0;                                   /* 3,3 */
    
    
    /* add the portion for the first constraint */
    values[1] += lambda[0] * (x[2] * x[3]);          /* 1,0 */
    
    values[3] += lambda[0] * (x[1] * x[3]);          /* 2,0 */
    values[4] += lambda[0] * (x[0] * x[3]);          /* 2,1 */
    
    values[6] += lambda[0] * (x[1] * x[2]);          /* 3,0 */
    values[7] += lambda[0] * (x[0] * x[2]);          /* 3,1 */
    values[8] += lambda[0] * (x[0] * x[1]);          /* 3,2 */
    
    /* add the portion for the second constraint */
    values[0] += lambda[1] * 2;                      /* 0,0 */
    
    values[2] += lambda[1] * 2;                      /* 1,1 */
    
    values[5] += lambda[1] * 2;                      /* 2,2 */
    
    values[9] += lambda[1] * 2;                      /* 3,3 */
  }
  
  return TRUE;
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


