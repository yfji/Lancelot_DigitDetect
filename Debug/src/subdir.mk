################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/DigitDetector.cpp \
../src/MLP.cpp \
../src/TwoLayerNNFaster.cpp \
../src/main.cpp \
../src/util.cpp \
../src/lancelot/CharacterRecognition.cpp

OBJS += \
./src/DigitDetector.o \
./src/MLP.o \
./src/TwoLayerNNFaster.o \
./src/main.o \
./src/util.o \
./src/lancelot/CharacterRecognition.o

CPP_DEPS += \
./src/DigitDetector.d \
./src/MLP.d \
./src/TwoLayerNNFaster.d \
./src/main.d \
./src/util.d \
./src/lancelot/CharacterRecognition.d


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I /home/jyf/Packages/OpenCV2.4.10/build/include -I/usr/local/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<" -std=c++11
	@echo 'Finished building: $<'
	@echo ' '


