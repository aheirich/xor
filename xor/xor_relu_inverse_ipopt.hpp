//
//  xor_relu_inverse_ipopt.hpp
//
//  Created by Heirich, Alan on 1/19/18.
//  Copyright © 2018 Heirich, Alan. All rights reserved.
//

#ifndef xor_relu_inverse_ipopt_hpp
#define xor_relu_inverse_ipopt_hpp

#include <stdio.h>
#include "IpStdCInterface.h"
#include "IpIpoptApplication.hpp"

#define FIVE_CONSTRAINTS 0
#define USE_EXTENDED_FLOAT 1
#define INITIALIZE_AT_FIXED_POINT 1

#define CASE_11

const unsigned numInputUnits = 2;
const unsigned numHiddenUnits = 16;
const unsigned numOutputUnits = 2;
const unsigned numActivations = numInputUnits + numHiddenUnits + numOutputUnits;
const unsigned numAlpha = numInputUnits + numHiddenUnits;
const unsigned numUnknowns = numActivations + numAlpha;
#if FIVE_CONSTRAINTS
const unsigned constraintsPerAlpha = 5;
#else
const unsigned constraintsPerAlpha = 3;
#endif
const unsigned numConstraints = constraintsPerAlpha * numAlpha;

/* This is an example how user_data can be used. */
struct MyUserData
{
  Number ao_target[numOutputUnits];
};

Bool eval_f(Index n, Number* x, Bool new_x,
            Number* obj_value, UserDataPtr user_data);

Bool eval_grad_f(Index n, Number* x, Bool new_x,
                 Number* grad_f, UserDataPtr user_data);

Bool eval_g(Index n, Number* x, Bool new_x,
            Index m, Number* g, UserDataPtr user_data);

Bool eval_jac_g(Index n, Number *x, Bool new_x,
                Index m, Index nele_jac,
                Index *iRow, Index *jCol, Number *values,
                UserDataPtr user_data);

Bool eval_h(Index n, Number *x, Bool new_x, Number obj_factor,
            Index m, Number *lambda, Bool new_lambda,
            Index nele_hess, Index *iRow, Index *jCol,
            Number *values, UserDataPtr user_data);

Bool intermediate_cb(Index alg_mod, Index iter_count, Number obj_value,
                     Number inf_pr, Number inf_du, Number mu, Number d_norm,
                     Number regularization_size, Number alpha_du,
                     Number alpha_pr, Index ls_trials, UserDataPtr user_data);


#endif /* xor_relu_inverse_ipopt_hpp */
