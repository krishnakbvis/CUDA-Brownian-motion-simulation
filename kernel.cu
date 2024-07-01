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

    if (i < N && j < N && i != j)
    {
        double dx = xPos[i] - xPos[j];
        double dy = yPos[i] - yPos[j];
        double sep = sqrt(dx * dx + dy * dy);

        // Avoid division by zero and self-interaction
        if (sep > 1e-9) {  // Use a small threshold to avoid zero division
            double invr7 = 1.0 / pow(sep, 7);
            double invr13 = 1.0 / pow(sep, 13);

            // Calculate force
            double force = 4 * epsilon * ((A * pow(sigma[i], 6) * invr7) - (B * pow(sigma[i], 12) * invr13));
            forceX[i * N + j] = -(dx / sep) * force;
            forceY[i * N + j] = -(dy / sep) * force;
        }
        else {
            forceX[i * N + j] = 0.0;
            forceY[i * N + j] = 0.0;
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



__global__ void integratePositions(int count, double* dev_xPosMatrix, double* dev_yPosMatrix, double* xPos, double* yPos, double* xVel, double* yVel, double* accX, double* accY, int N, double timeStep, double* radii, int boxwidth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (count % 100 == 0) {
            int row = count / 100;
            dev_xPosMatrix[row * N + i] = xPos[i];
            dev_yPosMatrix[row * N + i] = yPos[i];
        }

        xPos[i] += xVel[i] * timeStep + 0.5 * accX[i] * timeStep * timeStep;
        yPos[i] += yVel[i] * timeStep + 0.5 * accY[i] * timeStep * timeStep;

        // Handle boundary conditions after position update
        if (xPos[i] - radii[i] <= 0 || xPos[i] + radii[i] >= boxwidth) {
            xVel[i] = -xVel[i];
        }
        if (yPos[i] - radii[i] <= 0 || yPos[i] + radii[i] >= boxwidth) {
            yVel[i] = -yVel[i];
        }
    }
}


__global__ void integrateVelocities(double* xVel, double* yVel, double* oldAccX, double* oldAccY, double* newAccX, double* newAccY, int N, double timeStep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        xVel[i] += 0.5 * (newAccX[i] + oldAccX[i]) * timeStep;
        yVel[i] += 0.5 * (newAccY[i] + oldAccY[i]) * timeStep;

        oldAccX[i] = newAccX[i];
    }
}



// Function to compute accelerations
cudaError_t computeAccelerations(double* dev_xPos, double* dev_yPos, double* dev_masses, double* dev_accX, double* dev_accY, double* dev_sigma,
    const unsigned int N, const double A, const double B, const double epsilon, const double timeStep)
{
    double* dev_forceX;
    double* dev_forceY;
    cudaMalloc((void**)&dev_forceX, N * sizeof(double));
    cudaMalloc((void**)&dev_forceY, N * sizeof(double));

    // Define the grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel on the GPU
    computeForces <<<blocksPerGrid, threadsPerBlock>>> (dev_forceX, dev_forceY, dev_xPos, dev_yPos,
        N, A, B, dev_sigma, epsilon);
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    // Aggregate accelerations
    aggregateAccelerations <<<blocksPerGrid, threadsPerBlock.x >>> (dev_forceX, dev_forceY, dev_accX, dev_accY, dev_masses, N);
    cudaStatus = cudaDeviceSynchronize();

    // Free GPU buffers
    cudaFree(dev_forceX);
    cudaFree(dev_forceY);

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


void printMatrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i*cols + j]);
        }
        printf("\n");
    }
}





int main()
{
    const int N = 10;
    const double epsilon = 0.1;
    const double A = 1;
    const double B = 1;
    const double timeStep = 1E-4;
    const int runTime = 10;
    const int iterations = runTime / timeStep;
    const int boxwidth = 25;

    // Allocate memory
    double* xPos = (double*)malloc(N * sizeof(double));
    double* yPos = (double*)malloc(N * sizeof(double));
    double* xVel = (double*)malloc(N * sizeof(double));
    double* yVel = (double*)malloc(N * sizeof(double));
    double* masses = (double*)malloc(N * sizeof(double));
    double* sigma = (double*)malloc(N * sizeof(double));
    double* radii = (double*)malloc(N * sizeof(double));
    double* xPositionMatrix = (double*)malloc(runTime*100 * N * sizeof(double));
    double* yPositionMatrix = (double*)malloc(runTime*100 * N * sizeof(double));

    // Initialize positions, velocities, etc.
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        masses[i] = 1;
        xPos[i] = (double)rand() / (double)(RAND_MAX / 20);
        yPos[i] = (double)rand() / (double)(RAND_MAX / 20);
        xVel[i] = (double)rand() / (double)(RAND_MAX / 4);
        yVel[i] = (double)rand() / (double)(RAND_MAX / 4);
        radii[i] = 0.3;
        sigma[i] = 0.3/pow(2,1/6);
    }
    masses[N / 2] = 1; // Brownian particle
    radii[N / 2] = 0.7;

    // Allocate device memory
    double* dev_xPos, * dev_yPos, * dev_xVel, * dev_yVel, * dev_accX, * dev_accY;
    double* dev_newaccX, * dev_newaccY, * dev_sigma, * dev_masses, * dev_radii;
    double* dev_xmat, * dev_ymat;

    cudaMalloc((void**)&dev_xPos, N * sizeof(double));
    cudaMalloc((void**)&dev_yPos, N * sizeof(double));
    cudaMalloc((void**)&dev_xVel, N * sizeof(double));
    cudaMalloc((void**)&dev_yVel, N * sizeof(double));
    cudaMalloc((void**)&dev_accX, N * sizeof(double));
    cudaMalloc((void**)&dev_accY, N * sizeof(double));
    cudaMalloc((void**)&dev_newaccX, N * sizeof(double));
    cudaMalloc((void**)&dev_newaccY, N * sizeof(double));
    cudaMalloc((void**)&dev_sigma, N * sizeof(double));
    cudaMalloc((void**)&dev_masses, N * sizeof(double));
    cudaMalloc((void**)&dev_radii, N * sizeof(double));
    cudaMalloc((void**)&dev_xmat, runTime*100 * N * sizeof(double));
    cudaMalloc((void**)&dev_ymat, runTime*100 * N * sizeof(double));

    // Copy data to device
    cudaMemcpy(dev_xPos, xPos, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_yPos, yPos, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_xVel, xVel, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_yVel, yVel, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sigma, sigma, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_masses, masses, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_radii, radii, N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Compute initial accelerations
    computeAccelerations(dev_xPos, dev_yPos, dev_masses, dev_accX, dev_accY, dev_sigma, N, A, B, epsilon, timeStep);


    // Main loop
    for (int count = 0; count < iterations; count++) {
        integratePositions << <blocksPerGrid, threadsPerBlock >> > (count, dev_xmat, dev_ymat, dev_xPos, dev_yPos, 
            dev_xVel, dev_yVel, dev_accX, dev_accY, N, timeStep, dev_radii, boxwidth);
        cudaDeviceSynchronize();

        // Compute new accelerations after positions are updated
        computeAccelerations(dev_xPos, dev_yPos, dev_masses, dev_newaccX, dev_newaccY, dev_sigma, N, A, B, epsilon, timeStep);
        cudaDeviceSynchronize();

        // Update velocities using old and new accelerations
        integrateVelocities << <blocksPerGrid, threadsPerBlock >> > (dev_xVel, dev_yVel, dev_accX, dev_accY, 
            dev_newaccX, dev_newaccY, N, timeStep);
        cudaDeviceSynchronize();

    }


    // Copy results back to host
    cudaMemcpy(xPositionMatrix, dev_xmat, runTime*100 * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(yPositionMatrix, dev_ymat, runTime*100 * N * sizeof(double), cudaMemcpyDeviceToHost);

    writeMatrixToFile(xPositionMatrix, runTime * 100, N, "xPositionMatrix.csv");
    writeMatrixToFile(yPositionMatrix, runTime * 100, N, "yPositionMatrix.csv");



    // Free memory
    free(xPos);
    free(yPos);
    free(xVel);
    free(yVel);
    free(masses);
    free(sigma);
    free(radii);
    free(xPositionMatrix);
    free(yPositionMatrix);

    cudaFree(dev_xPos);
    cudaFree(dev_yPos);
    cudaFree(dev_xVel);
    cudaFree(dev_yVel);
    cudaFree(dev_accX);
    cudaFree(dev_accY);
    cudaFree(dev_newaccX);
    cudaFree(dev_newaccY);
    cudaFree(dev_sigma);
    cudaFree(dev_masses);
    cudaFree(dev_radii);
    cudaFree(dev_xmat);
    cudaFree(dev_ymat);

    return 0;
}
