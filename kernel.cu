#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <chrono>
//Так делать плохо
#include <iostream>

# define M_PI 3.14159265358979323846

double Ax = -0.353, Bx = 0.353, Ay = 0.3, By = 0.3, C = (3 * M_PI) / 8, a2 = (3 * M_PI) / 2;
double p = 2000.0, m = 100.0, g = 10.0, v = 0.0;

const int N = 5;
const double delta_r = 0.005, delta_t = 0.01;
std::chrono::time_point<std::chrono::steady_clock> begin, end;


cudaError_t findWithCuda(double* X0, double* constatns);
void findWithoutCuda(double* X0, double* constants);
__global__ void FiAndCalc(double x1, double x2, double f1, double f2, double y, double* X0, double* X1, double* constatns, double* Fs);
void FiAndCalcS(double x1, double x2, double f1, double f2, double y, double* X0, double* X1, double* constatns, int i);

__device__
void getFX1(double* Fs, double* X0, double* constatns) {
    Fs[0] = (X0[0] + X0[2] * cos(3 * M_PI / 2 - X0[3]) - constatns[0]);
}

__device__
void getFX2(double* Fs, double* X0, double* constatns) {
    Fs[1] = (X0[1] + X0[2] * cos(3 * M_PI / 2 + X0[4]) - constatns[1]);
}

__device__
void getFY(double* Fs, double* X0, double* constatns) {
    Fs[2] = (X0[2] + X0[2] * sin(3 * M_PI / 2 - X0[3]) - constatns[2]);
}

__device__
void getFF1(double* Fs, double* X0, double* constatns) {
    Fs[3] = ((X0[3] + X0[4]) * X0[2] + (X0[1] - X0[0]) - constatns[4]);
}

__device__
void getFF2(double* Fs, double* X0, double* constatns) {
    Fs[4] = (X0[2] + X0[2] * sin(3 * M_PI / 2 + X0[4]) - constatns[3]);
}

int main()
{
    FILE* ouf = fopen("multi_treads.csv", "w");
    fprintf(ouf, "step,x1,x2,y,f1,f2,Ax,Ay,Bx,By,C,time\n");

    double constats[] = { Ax, Bx, Ay, By, C, delta_r };
    double X0[] = { -0.1, 0.1, 2.0, 2.0, 0.0 };

    int step = 0;
    for (double t = 0; t <= 2.5; t += delta_t, step += 1) {
        clock_t start = clock();

        cudaError_t cudaStatus = findWithCuda(X0, constats);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }

        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }

        clock_t end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Time spend = %lf", time_taken);

        constats[2] += v * delta_t;
        constats[3] = constats[2];
        v += (p * (X0[1] - X0[0]) - m * g) / m * delta_t;
        

        printf(" steps_done = %d ", step);
        fprintf(ouf, "%d, ", step);
        for (int i = 0; i < 5; ++i) {
            fprintf(ouf, "%f, ", X0[i]);
        }
        fprintf(ouf, "%f, %f, %f, %f, %f, %f\n", Ax, Ay, Bx, By, C, time_taken);
        fflush(ouf);
    }
    fclose(ouf);


    FILE* ouf_2 = fopen("single_tread.csv", "w");
    fprintf(ouf_2, "step,x1,x2,y,f1,f2,Ax,Ay,Bx,By,C,time\n");
    double X0_[] = { -0.1, 0.1, 2.0, 2.0, 0.0 };
    double constats_[] = { Ax, Bx, Ay, By, C, delta_r };

    step = 0;
    v = 0;
    for (double t = 0; t <= 2.5; t += delta_t, step += 1) {
        clock_t start = clock();
        findWithoutCuda(X0_, constats_);
        clock_t end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Time spend = %lf", time_taken);

        constats_[2] += v * delta_t;
        constats_[3] = constats_[2];
        v += (p * (X0_[1] - X0_[0]) - m * g) / m * delta_t;


        fprintf(ouf_2, "%d, ", step);
        for (int i = 0; i < 5; ++i) {
            if (i == 2) fprintf(ouf_2, "%f, ", X0_[4]);
            else if (i == 4) fprintf(ouf_2, "%f, ", X0_[2]);
            else fprintf(ouf_2, "%f, ", X0_[i]);
        }
        fprintf(ouf_2, "%f, %f, %f, %f, %f, %f\n", Ax, Ay, Bx, By, C, time_taken);
        fflush(ouf_2);
    }
    fclose(ouf_2);

    return 0;
}

void findWithoutCuda(double* X0, double* constants) {
    double* dev_X0 = (double*)malloc(5 * sizeof(double));
    for (int i = 0; i < 5; ++i)
        dev_X0[i] = X0[i];
    double* dev_constants = constants;

    long long steps = 0;
    while (true)
    {
        int flag = 0;
        ++steps;
        double dev_X1[] = { 0.0, 0.0, 0.0, 0.0, 0.0 };

        begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 5; ++i) {
            FiAndCalcS(X0[0], X0[1], X0[2], X0[3], X0[4], dev_X0, dev_X1, dev_constants, i);
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Clean Time without Cuda: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << " nanos\n";

        if ((fabs(dev_X0[0] - dev_X1[0]) <= 1e-8 &&
            fabs(dev_X0[1] - dev_X1[1]) <= 1e-8 &&
            fabs(dev_X0[2] - dev_X1[2]) <= 1e-8 &&
            fabs(dev_X0[3] - dev_X1[3]) <= 1e-8 &&
            fabs(dev_X0[4] - dev_X1[4]) <= 1e-8) || steps > 200000) {
            flag = 1;
        }

        for (int i = 0; i < 5; ++i) {
            X0[i] = dev_X1[i];
            dev_X0[i] = dev_X1[i];
        }

        if (flag) {
            printf("\nsteps=%d\n", steps);
            printf("x1=%lf\n", X0[0]);
            printf("x2=%lf\n", X0[1]);
            printf("y=%lf\n", X0[4]);
            printf("f1=%lf\n", X0[3]);
            printf("f2=%lf\n", X0[2]);
            free(dev_X0);
            break;
        }

    }
}


cudaError_t findWithCuda(double* X0, double* constants)
{
    double* dev_X0 = 0;
    double* dev_X1 = 0;
    double* dev_constants = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for data.
    cudaStatus = cudaMalloc((void**)&dev_X0, 5 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_X1, 5 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_X0, X0, 5 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_constants, 6 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_constants, constants, 6 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    double* Fs = 0;
    cudaStatus = cudaMalloc((void**)&Fs, 5 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    long long steps = 0;
    while (true)
    {
        int flag = 0;
        ++steps;
        double X1[] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
        cudaStatus = cudaMemcpy(dev_X1, X1, 5 * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        begin = std::chrono::steady_clock::now();
        // Launch a kernel on the GPU with one thread for each element.
        FiAndCalc<< <1, N>> >(X0[0], X0[1], X0[2], X0[3], X0[4], dev_X0, dev_X1, dev_constants, Fs);

        end = std::chrono::steady_clock::now();
        std::cout << "Clean Time with cuda: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << " nanos\n";

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        cudaStatus = cudaMemcpy(X1, dev_X1, 5 * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }



        if ((fabs(X0[0] - X1[0]) <= 1e-8 &&
            fabs(X0[1] - X1[1]) <= 1e-8 &&
            fabs(X0[2] - X1[2]) <= 1e-8 &&
            fabs(X0[3] - X1[3]) <= 1e-8 &&
            fabs(X0[4] - X1[4]) <= 1e-8) || steps > 1000000) {
            flag = 1;
        }

        for (int i = 0; i < 5; ++i) {
            X0[i] = X1[i];
        }

        cudaStatus = cudaMemcpy(dev_X0, dev_X1, N * sizeof(double), cudaMemcpyDeviceToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
        if (flag) {
            printf("\nsteps=%d\n", steps);
            printf("x1=%lf\n", X0[0]);
            printf("x2=%lf\n", X0[1]);
            printf("y=%lf\n", X0[2]);
            printf("f1=%lf\n", X0[3]);
            printf("f2=%lf\n", X0[4]);
            break;
        }
    }

Error:
    cudaFree(dev_X0);
    cudaFree(dev_X1);
    cudaFree(Fs);
    
    return cudaStatus;
}


__global__ void FiAndCalc(double x1, double x2, double f1, double f2, double y, double* X0, double* X1, double* constatns, double* Fs) {
    int i = blockIdx.x* blockDim.x + threadIdx.x;
    if (i == 0) getFX1(Fs, X0, constatns);
    else if (i == 1) getFX2(Fs, X0, constatns);
    else if (i == 2) getFY(Fs, X0, constatns);
    else if (i == 3) getFF1(Fs, X0, constatns);
    else if (i == 4) getFF2(Fs, X0, constatns);

    X1[i] = X0[i] - constatns[5] * Fs[i];
}

void FiAndCalcS(double x1, double x2, double f1, double f2, double y, double* X0, double* X1, double* constatns, int i) {
    double Fi = 0.0;
    if (i == 0) Fi = x1 + y * cos(3 * M_PI / 2 - f1) - constatns[0];
    if (i == 1) Fi = x2 + y * cos(3 * M_PI / 2 + f2) - constatns[1];
    if (i == 2) Fi = y + y * sin(3 * M_PI / 2 - f1) - constatns[2];
    if (i == 3) Fi = (f1 + f2) * y + (x2 - x1) - constatns[4];
    if (i == 4) Fi = y + y * sin(3 * M_PI / 2 + f2) - constatns[3];

    X1[i] = X0[i] - constatns[5] * Fi;
}
