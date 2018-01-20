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

// input weights to hidden layer
Number W1[2][10] = {{ 0.9145419,  -0.5964824,  -0.642739,   -0.34978598,  0.41890582,  0.22138357,
  -0.08724618, -0.17340243, -0.06329376,  0.45263577},
  {-0.88043803,  0.70016438,  0.94026381,  0.19096281, -0.66918135,  0.26964825,
    -0.33756,    -0.1920374,  -0.11555147,  0.92417163 }};
// hidden layer biases
Number b1[10] = { -1.29862165e-08,  -2.89841706e-10,  -3.04766568e-09,  -1.96804002e-01,
  -3.54976351e-08,   6.22820675e-01,   0.00000000e+00,   0.00000000e+00,
  0.00000000e+00,  -4.52658087e-01 };
// hidden weights to output layer
Number W2[10][2] = {{ 0.99346685, -0.81308043},
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
Number b2[2] = {-0.049427,    0.62176669};






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
  Index nele_jac = 8;
  /* Number of nonzeros in the Hessian of the Lagrangian (lower or
   upper triangual part only) */
  Index nele_hess = 10;
  /* indexing style for matrices */
  Index index_style = 0; /* C-style; start counting of rows and column
                          indices at 0 */
  
  /* our user data for the function evalutions. */
  struct MyUserData user_data;
  
  /* set the number of variables and allocate space for the bounds */
  n=4;
  x_L = (Number*)malloc(sizeof(Number)*n);
  x_U = (Number*)malloc(sizeof(Number)*n);
  /* set the values for the variable bounds */
  for (i=0; i<n; i++) {
    x_L[i] = 1.0;
    x_U[i] = 5.0;
  }
  
  /* set the number of constraints and allocate space for the bounds */
  m=2;
  g_L = (Number*)malloc(sizeof(Number)*m);
  g_U = (Number*)malloc(sizeof(Number)*m);
  /* set the values of the constraint bounds */
  g_L[0] = 25;
  g_U[0] = 2e19;
  g_L[1] = 40;
  g_U[1] = 40;
  
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
  user_data.g_offset[0] = 0.;
  user_data.g_offset[1] = 0.;
  
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
Bool eval_f(Index n, Number* x, Bool new_x,
            Number* obj_value, UserDataPtr user_data)
{
  assert(n == 4);
  
  *obj_value = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
  
  return TRUE;
}

Bool eval_grad_f(Index n, Number* x, Bool new_x,
                 Number* grad_f, UserDataPtr user_data)
{
  assert(n == 4);
  
  grad_f[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
  grad_f[1] = x[0] * x[3];
  grad_f[2] = x[0] * x[3] + 1;
  grad_f[3] = x[0] * (x[0] + x[1] + x[2]);
  
  return TRUE;
}




Bool eval_g(Index n, Number* x, Bool new_x,
            Index m, Number* g, UserDataPtr user_data)
{
  struct MyUserData* my_data = (struct MyUserData*)user_data;
  
  assert(n == 4);
  assert(m == 2);
  
  g[0] = x[0] * x[1] * x[2] * x[3] + my_data->g_offset[0];
  g[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] + my_data->g_offset[1];
  
  return TRUE;
}




Bool eval_jac_g(Index n, Number *x, Bool new_x,
                Index m, Index nele_jac,
                Index *iRow, Index *jCol, Number *values,
                UserDataPtr user_data)
{
  if (values == NULL) {
    /* return the structure of the jacobian */
    
    /* this particular jacobian is dense */
    iRow[0] = 0;
    jCol[0] = 0;
    iRow[1] = 0;
    jCol[1] = 1;
    iRow[2] = 0;
    jCol[2] = 2;
    iRow[3] = 0;
    jCol[3] = 3;
    iRow[4] = 1;
    jCol[4] = 0;
    iRow[5] = 1;
    jCol[5] = 1;
    iRow[6] = 1;
    jCol[6] = 2;
    iRow[7] = 1;
    jCol[7] = 3;
  }
  else {
    /* return the values of the jacobian of the constraints */
    
    values[0] = x[1]*x[2]*x[3]; /* 0,0 */
    values[1] = x[0]*x[2]*x[3]; /* 0,1 */
    values[2] = x[0]*x[1]*x[3]; /* 0,2 */
    values[3] = x[0]*x[1]*x[2]; /* 0,3 */
    
    values[4] = 2*x[0];         /* 1,0 */
    values[5] = 2*x[1];         /* 1,1 */
    values[6] = 2*x[2];         /* 1,2 */
    values[7] = 2*x[3];         /* 1,3 */
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


