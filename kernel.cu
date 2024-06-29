#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel to compute force matrix
// Kernel to compute force matrix
__global__ void computeForces(double* forceX, double* forceY, double* xPos, double* yPos,
    int N, int A, int B, double* sigma, const double epsilon)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N)
    {
        double dx = xPos[i] - xPos[j];
        double dy = yPos[i] - yPos[j];
        double sep = sqrt(dx * dx + dy * dy);

        // Avoid division by zero and self-interaction
        if (sep == 0) {
            forceX[i * N + j] = 0.0;
            forceY[i * N + j] = 0.0;
        }
        else {
            double invr7 = 1.0 / (pow(sep, 7));
            double invr13 = 1.0 / (pow(sep, 13));

            // Calculate force avoiding NaN
            double force = 4 * epsilon * (A * pow(sigma[i], 6) * invr7 - B * pow(sigma[i], 12) * invr13);
            forceX[i * N + j] = (dx / sep) * force;
            forceY[i * N + j] = (dy / sep) * force;
        }
    }
}

// Kernel to aggregate accelerations
// Kernel to aggregate accelerations
__global__ void aggregateAccelerations(double* forceX, double* forceY, double* accX, double* accY, const double* masses, int N)
{
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIdx < N)
    {
        double sumX = 0;
        double sumY = 0;
        for (int col = 0; col < N; ++col)
        {
            sumX += forceX[rowIdx * N + col];
            sumY += forceY[rowIdx * N + col];
        }

        // Avoid NaN in acceleration calculation
        if (masses[rowIdx] != 0.0) {
            accX[rowIdx] = sumX / masses[rowIdx];
            accY[rowIdx] = sumY / masses[rowIdx];
        }
        else {
            accX[rowIdx] = 0.0;
            accY[rowIdx] = 0.0;
        }
    }
}


// Kernel to perform numerical integration
__global__ void numericalIntegrate(int row, double* xPositionMatrix, double* yPositionMatrix, double* xPos, double* yPos, double* xVel,
    double* yVel, double* accX, double* accY, int N, double timeStep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        xPositionMatrix[(row % 100) * N + i] = xPos[i];
        yPositionMatrix[(row % 100) * N + i] = yPos[i];

        xVel[i] += accX[i] * timeStep;
        yVel[i] += accY[i] * timeStep;

        xPos[i] += xVel[i] * timeStep;
        yPos[i] += yVel[i] * timeStep;
    }
}

// Function to compute accelerations
cudaError_t computeAccelerations(double* xPos, double* yPos, double* masses, double* accX, double* accY, double* sigma,
    const unsigned int N, const double A, const double B, const double epsilon, double timeStep)
{
    double* dev_forceX;
    double* dev_forceY;
    double* dev_xPos;
    double* dev_yPos;
    double* dev_accX;
    double* dev_accY;
    double* dev_masses;
    double* dev_sigma;

    double* forceX = (double*) malloc(N * N * sizeof(double));
    double* forceY = (double*)malloc(N * N * sizeof(double));

    // Allocate GPU buffers for vectors
    cudaMalloc((void**)&dev_forceX, N * N * sizeof(double));
    cudaMalloc((void**)&dev_forceY, N * N * sizeof(double));
    cudaMalloc((void**)&dev_accX, N * sizeof(double));
    cudaMalloc((void**)&dev_accY, N * sizeof(double));
    cudaMalloc((void**)&dev_xPos, N * sizeof(double));
    cudaMalloc((void**)&dev_yPos, N * sizeof(double));
    cudaMalloc((void**)&dev_masses, N * sizeof(double));
    cudaMalloc((void**)&dev_sigma, N * sizeof(double));

    // Copy input vectors from host memory to GPU buffers
    cudaMemcpy(dev_xPos, xPos, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_yPos, yPos, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_masses, masses, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sigma, sigma, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_forceX, forceX, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_forceY, forceY, N * N * sizeof(double), cudaMemcpyHostToDevice);



    // Define the grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel on the GPU
    computeForces << <blocksPerGrid, threadsPerBlock >> > (dev_forceX, dev_forceY, dev_xPos, dev_yPos,
        N, A, B, dev_sigma, epsilon);

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    // Aggregate accelerations
    aggregateAccelerations << <(N + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock.x >> > (dev_forceX, dev_forceY, dev_accX, dev_accY, dev_masses, N);
    cudaStatus = cudaDeviceSynchronize();
    // Copy output vectors from GPU buffer to host memory
    cudaMemcpy(accX, dev_accX, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(accY, dev_accY, N * sizeof(double), cudaMemcpyDeviceToHost);


//     Print force matrices for debugging
    printf("accX:\n");
    for (int j = 0; j < N; j++) {
        printf("%f ", accX[j]);
    }
    

    printf("accY:\n");
    for (int j = 0; j < N; j++) {
        printf("%f ", accY[j]);
    }


    // Free GPU buffers
    cudaFree(dev_forceX);
    cudaFree(dev_forceY);
    cudaFree(dev_xPos);
    cudaFree(dev_yPos);
    cudaFree(dev_accX);
    cudaFree(dev_accY);
    cudaFree(dev_masses);
    cudaFree(dev_sigma);

    return cudaStatus;
}

// Main function
int main()
{
    const int N = 100;
    double epsilon = 5;
    double A = 3;
    double B = 1;
    int count = 0;
    double timeStep = pow(10, -4);
    int iterations = 20 / timeStep;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Allocate memory for vectors
    double* xPos = (double*)malloc(N * sizeof(double));
    double* yPos = (double*)malloc(N * sizeof(double));
    double* xVel = (double*)malloc(N * sizeof(double));
    double* yVel = (double*)malloc(N * sizeof(double));
    double* masses = (double*)malloc(N * sizeof(double));
    double* forceX = (double*)malloc(N * N * sizeof(double));
    double* forceY = (double*)malloc(N * N * sizeof(double));
    double* sigma = (double*)malloc(N * sizeof(double));
    double* radii = (double*)malloc(N * sizeof(double));
    double* accY = (double*)malloc(N * sizeof(double));
    double* accX = (double*)malloc(N * sizeof(double));
    double* xPositionMatrix = (double*)malloc(iterations * N * sizeof(double));
    double* yPositionMatrix = (double*)malloc(iterations * N * sizeof(double));

    double* dev_xPositionMatrix;
    double* dev_yPositionMatrix;
    cudaMalloc((void**)&dev_xPositionMatrix, iterations * N * sizeof(double));
    cudaMalloc((void**)&dev_yPositionMatrix, iterations * N * sizeof(double));
    
    srand(time(NULL)); // Initialize random number generator

    // Initialize velocities, positions, and masses
    for (int i = 0; i < N; i++) {
        masses[i] = 1;
        xPos[i] = rand() % 50 - 20;
        yPos[i] = rand() % 50 - 20;
        xVel[i] = rand() % 4 - 2;
        yVel[i] = rand() % 4 - 2;
        radii[i] = 0.3;
        sigma[i] = 1;
    }
    masses[N / 2] = 1000; // Set Brownian particle mass
    radii[N / 2] = 0.7;




    // Compute initial accelerations
    computeAccelerations(xPos, yPos, masses, accX, accY, sigma, N, A, B, epsilon, timeStep);

    // Perform numerical integration
    numericalIntegrate << <blocksPerGrid, threadsPerBlock >> > (count, dev_xPositionMatrix, dev_yPositionMatrix, xPos, yPos, xVel, yVel, accX, accY, N, timeStep);
    cudaDeviceSynchronize();

    // Copy position matrices back to host
    cudaMemcpy(xPositionMatrix, dev_xPositionMatrix, iterations * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(yPositionMatrix, dev_yPositionMatrix, iterations * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Free allocated memory
    free(xPos);
    free(yPos);
    free(xVel);
    free(yVel);
    free(masses);
    free(forceX);
    free(forceY);
    free(sigma);
    free(radii);
    free(accY);
    free(accX);

    cudaFree(dev_xPositionMatrix);
    cudaFree(dev_yPositionMatrix);

    return 0;
}
