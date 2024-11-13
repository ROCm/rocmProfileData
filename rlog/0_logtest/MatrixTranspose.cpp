/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>

// hip header file
#include "hip/hip_runtime.h"
#include "../../rpd_tracer/Utility.h"
//#include "roctracer/roctx.h"
#include "/usr/local/include/rlog.h"


#define WIDTH 1024


#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y];
}

//bool isLogging = false;
//void rlog_callback() {
//    isLogging = rlog::isActive();
//    fprintf(stderr, "rlog_callback()  %s\n", isLogging ? "True" : "False");
//}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

namespace rlog {

    bool isLogging;
    void rlog_callback_function() {
        isLogging = rlog::isActive();
    }

    class Client {
    public:
        Client() {
            fprintf(stderr, "Client ++++++++++++++++\n");
            rlog::init();
            rlog::registerActiveCallback(&rlog_callback_function);
            rlog::setDefaultDomain("MIOpen");
            rlog::setDefaultCategory("");
        }
    };
    Client client;
} // namespace rlog


int main() {
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    int i;
    int errors;

    Matrix = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix = (float*)malloc(NUM * sizeof(float));
    cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

    // Memory transfer from host to device
    hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);


    // Fire up logging
    //rlog::init();
    //rlog::registerActiveCallback(&rlog_callback);
    //rlog::setDefaultDomain("MT");
    //rlog::setDefaultCategory("test");
    //if (isLogging)
    //    rlog::mark("test", "nocall", "noargs");

    // Warmup
    for (int i = 0 ; i < 100; ++i) {
    // Lauching kernel from host
    hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
                    gpuMatrix, WIDTH);
    }


    int count = 2000;

    timestamp_t t1, t2, t3;
    t1 = clocktime_ns();

    for (int i = 0 ; i < count; ++i) {
    // Lauching kernel from host
    hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
                    gpuMatrix, WIDTH);
    }
    t2 = clocktime_ns();

    hipDeviceSynchronize();

    t3 = clocktime_ns();

    fprintf(stderr, "hipLaunchKernel: %d in %f ms.  %f ns / call\n", count, (t2 - t1) / 1000000.0, 1.0*(t2-t1)/count);
    fprintf(stderr, "hipDeviceSynchronize: %f ms\n", (t3 - t2) / 1000000.0);


    // hip get/set
    int devid = 0;
    t1 = clocktime_ns();
    for (int i = 0 ; i < count; ++i) {
        hipGetDevice(&devid);
        hipGetDevice(&devid);
        hipGetDevice(&devid);
        hipGetDevice(&devid);
        hipSetDevice(devid);
        hipGetDevice(&devid);
        hipGetDevice(&devid);
        hipGetDevice(&devid);
        hipGetDevice(&devid);
        hipSetDevice(devid);
    }
    t2 = clocktime_ns();

    fprintf(stderr, "hipGetSetDevice: %d in %f ms.  %f ns / call\n", count * 10, (t2 - t1) / 1000000.0, 0.1*(t2-t1)/count);

    count = 2;
    count = atoi(rlog::getProperty("MT", "rangeCount", "10")); 

    // roctx static
    char buff[4096];
    sprintf(buff, "this is a medium size roctx message, %d", count);
    t1 = clocktime_ns();
    for (int i = 0 ; i < count; ++i) {
        //roctxRangePushA(buff);
        //roctxRangePop();
        if (rlog::isLogging) {
            rlog::rangePush("static", buff);
            rlog::rangePop();
        }
    }
    t2 = clocktime_ns();
    fprintf(stderr, "roxtx_static: %d in %f ms.  %f ns / call\n", count, (t2 - t1) / 1000000.0, 1.0*(t2-t1)/count);

    //roctx variable
    char *msg[count];
    for (int i = 0 ; i < count; ++i) {
	msg[i] = new char[4096];
        sprintf(msg[i], "this is a medium size roctx message, %d", i);
    }

    t1 = clocktime_ns();
    for (int i = 0 ; i < count; ++i) {
        //roctxRangePushA(msg[i]);
        //roctxRangePop();
        if (rlog::isLogging) {
            rlog::rangePush("variable", msg[i]);
            rlog::rangePop();
        }
    }
    t2 = clocktime_ns();
    fprintf(stderr, "roctx_variable: %d in %f ms.  %f ns / call\n", count, (t2 - t1) / 1000000.0, 1.0*(t2-t1)/count);

    // Memory transfer from device to host
    hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);

    // CPU MatrixTranspose computation
    matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    errors = 0;
    double eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
        if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
            errors++;
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }

    // free the resources on device side
    hipFree(gpuMatrix);
    hipFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

    return errors;
}
