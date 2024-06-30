#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

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
__global__ void integratePositions(int row, double* xPositionMatrix, double* yPositionMatrix, double* xPos, double* yPos, double* xVel,
    double* yVel, double* accX, double* accY, int N, double timeStep, double* radii, int boxwidth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        if (row % 100 == 0) {
            int r = row / 100;
            xPositionMatrix[(r) * N + i] = xPos[i];
            yPositionMatrix[(r) * N + i] = yPos[i];
        }

        xPos[i] += xVel[i] * timeStep + 0.5*accX[i]*timeStep*timeStep;
        yPos[i] += yVel[i] * timeStep + 0.5 * accY[i] * timeStep * timeStep;

        if ((xPos[i] - radii[i] <= 0) || (xPos[i] + radii[i] >= boxwidth)) {
            xVel[i] = -xVel[i];
        }
        if ((yPos[i] - radii[i] <= 0) || (yPos[i] + radii[i] >= boxwidth)) {
            yVel[i] = -yVel[i];
        }

    }
}


__global__ void integrateVelocities(double* xVel,
    double* yVel, double* oldAccX, double* oldAccY, double* newAccX, double* newAccY, int N, double timeStep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        xVel[i] = xVel[i] + 0.5 * (newAccX[i] + oldAccX[i]) * timeStep;
        yVel[i] = yVel[i] + 0.5 * (newAccY[i] + newAccY[i]) * timeStep;

        oldAccX[i] = newAccX[i];
        oldAccY[i] = newAccY[i];
    }
}


// Function to compute accelerations
cudaError_t computeAccelerations(double* dev_xPos, double* dev_yPos, double* masses, double* accX, double* accY, double* sigma,
    const unsigned int N, const double A, const double B, const double epsilon, double timeStep)
{
    double* dev_forceX;
    double* dev_forceY;
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


    // Free GPU buffers
    cudaFree(dev_forceX);
    cudaFree(dev_forceY);
    cudaFree(dev_masses);
    cudaFree(dev_sigma);

    return cudaStatus;
}


#include <stdio.h>

// Function to write a matrix to a CSV file
void writeMatrixToFile(double* matrix, int rows, int cols, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s for writing.\n", filename);
        return;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%f", matrix[i * cols + j]);
            if (j < cols - 1) fprintf(file, ","); // No trailing comma at the end of the row
        }
        fprintf(file, "\n"); // New line at the end of each row
    }
    fclose(file);
}

// Main function
int main()
{
    const int N = 200;
    double epsilon = 5;
    double A = 3;
    double B = 1;
    int count = 0;
    double t = 0;
    double runTime = 10;
    double timeStep = pow(10, -4);
    double boxwidth = 25;
    int iterations = runTime / timeStep;

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
    double* xPositionMatrix = (double*)malloc(runTime*100 * N * sizeof(double));
    double* yPositionMatrix = (double*)malloc(runTime*100 * N * sizeof(double));

    double* dev_xPositionMatrix;
    double* dev_yPositionMatrix;
    double* dev_xPos;
    double* dev_yPos;
    double* dev_xVel;
    double* dev_yVel;
    double* dev_accX;
    double* dev_accY;
    double* dev_radii;
    double* dev_newaccX;
    double* dev_newaccY;
    cudaMalloc((void**)&dev_xPositionMatrix, runTime*100 * N * sizeof(double));
    cudaMalloc((void**)&dev_yPositionMatrix, runTime*100 * N * sizeof(double));
    cudaMalloc((void**)&dev_xPos, N * sizeof(double));
    cudaMalloc((void**)&dev_yPos, N * sizeof(double));
    cudaMalloc((void**)&dev_xVel, N * sizeof(double));
    cudaMalloc((void**)&dev_yVel, N * sizeof(double));
    cudaMalloc((void**)&dev_accX, N * sizeof(double));
    cudaMalloc((void**)&dev_accY, N * sizeof(double));
    cudaMalloc((void**)&dev_radii, N * sizeof(double));
    cudaMalloc((void**)&dev_newaccX, N * sizeof(double));
    cudaMalloc((void**)&dev_newaccY, N * sizeof(double));




    srand(time(NULL)); // Initialize random number generator
    // Initialize velocities, positions, and masses
    for (int i = 0; i < N; i++) {
        masses[i] = 1;
        xPos[i] = (double) rand() / (double)(RAND_MAX / 20);
        yPos[i] = (double)rand() / (double)(RAND_MAX / 20);
        xVel[i] = (double)rand() / (double)(RAND_MAX / 4);
        yVel[i] = (double)rand() / (double)(RAND_MAX / 4);
        radii[i] = 0.3;
        sigma[i] = 1;
    }
    masses[N / 2] = 1000; // Set Brownian particle mass
    radii[N / 2] = 0.7;

    cudaMemcpy(dev_xPos, xPos, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_yPos, yPos, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_xVel, xVel, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_yVel, yVel, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_radii, radii, N * sizeof(double), cudaMemcpyHostToDevice);

    computeAccelerations(dev_xPos, dev_yPos, masses, dev_accX, dev_accY, sigma, N, A, B, epsilon, timeStep);



    for (count = 0; count < iterations; count++) {
        integratePositions<<<blocksPerGrid, threadsPerBlock>>>(count, dev_xPositionMatrix, dev_yPositionMatrix, dev_xPos, dev_yPos, dev_xVel, dev_yVel, dev_accX, dev_accY, N, timeStep, dev_radii, boxwidth);
        // Compute accelerations
        computeAccelerations (dev_xPos, dev_yPos, masses, dev_newaccX, dev_newaccY, sigma, N, A, B, epsilon, timeStep);
        // Perform numerical integration
        integrateVelocities << <blocksPerGrid, threadsPerBlock >> > (dev_xVel,
            dev_yVel, dev_accX, dev_accY, dev_newaccX, dev_newaccY, N, timeStep);
        cudaDeviceSynchronize();
        t += timeStep;
    }

    // Copy final position matrices back from device to host
    cudaMemcpy(xPositionMatrix, dev_xPositionMatrix, runTime*100 * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(yPositionMatrix, dev_yPositionMatrix, runTime*100 * N * sizeof(double), cudaMemcpyDeviceToHost);
    
    writeMatrixToFile(xPositionMatrix, runTime * 100, N, "xPositionMatrix.csv");
    writeMatrixToFile(yPositionMatrix, runTime * 100, N , "yPositionMatrix.csv");


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
