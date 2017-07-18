################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/SPS.cpp \
../src/generic.cpp \
../src/host.cpp \
../src/mathop.cpp \

#CPP_SRCS += ../src/mathop_avx.cpp \

CPP_SRCS += ../src/mathop_sse2.cpp \
../src/random.cpp \
../src/slic.cpp 

OBJS += \
./src/SPS.o \
./src/generic.o \
./src/host.o \
./src/mathop.o \

#OBJS += ./src/mathop_avx.o \

OBJS += ./src/mathop_sse2.o \
./src/random.o \
./src/slic.o 

CPP_DEPS += \
./src/SPS.d \
./src/generic.d \
./src/host.d \
./src/mathop.d \

#CPP_DEPS += ./src/mathop_avx.d \

CPP_DEPS += ./src/mathop_sse2.d \
./src/random.d \
./src/slic.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O3 -g3 -Wall -c -fmessage-length=0 -fopenmp -msse4 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


