# SPS
Sampled Photoconsistency Stereo

The code is currently formatted to run with sse4.
To enable AVX support there are several things that have to be done:

In mathop.cpp comment out the definition VL_DISABLE_AVX

in the makefile replace -msse4 with -mavx
in build/src/subdir.mk uncomment the mathop_avx lines in CPP_SRCS, OBJS and CPP_DEPS and replace -msse4 with -mavx

vlfeat has no code that supports AVX2 at the moment. SPS implements the NCC functions with AVX2 support. 
It is highly recommended to not enable AVX2 because there will be speed issues when switching from vlfeat AVX to AVX2.
To enable AVX2 for SPS uncomment USE_AVX2 definition and replace -msse4 with -mavx2 in the makefile and build/src/subdir.mk

The THREADS macro specifies the number of threads to use with OpenMP. 

To compile under windows you need to follow the steps of installing openCV. Then use cmake to build the windows version of the code. 
