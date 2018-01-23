# Makefile

IPOPTLIB=${IPOPTDIR}/build/Ipopt/src/Interfaces/.libs/libipopt.so

xor_relu_inverse_ipopt: xor_relu_inverse_ipopt.cpp xor_relu_inverse_ipopt.hpp
	g++ -o xor_relu_inverse_ipopt -I${IPOPTDIR}/Ipopt/src/Interfaces/ -I${IPOPTDIR}/Ipopt/src/Common/ -I${IPOPTDIR}/Ipopt/src/LinAlg/ xor_relu_inverse_ipopt.cpp ${IPOPTLIB}
