#include <iostream>
#include <cuda_runtime.h>
#include "particle_filter.cuh"

#define N_PARTICLES 50000
#define N_LANDMARKS 4

int main() {
    int N = N_PARTICLES;
    int L = N_LANDMARKS;

    float *d_x, *d_y, *d_theta, *d_weights;
    float *d_x_new, *d_y_new, *d_theta_new;
    curandState* d_states;
    Landmark* d_landmarks;
    float* d_z;

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_theta, N * sizeof(float));
    cudaMalloc(&d_weights, N * sizeof(float));
    cudaMalloc(&d_x_new, N * sizeof(float));
    cudaMalloc(&d_y_new, N * sizeof(float));
    cudaMalloc(&d_theta_new, N * sizeof(float));
    cudaMalloc(&d_states, N * sizeof(curandState));
    cudaMalloc(&d_landmarks, L * sizeof(Landmark));
    cudaMalloc(&d_z, L * sizeof(float));

    Landmark h_landmarks[N_LANDMARKS] = {{0,0}, {5,0}, {5,5}, {0,5}};
    float h_z[N_LANDMARKS] = {3.6f, 2.5f, 6.1f, 5.2f};

    cudaMemcpy(d_landmarks, h_landmarks, L * sizeof(Landmark), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, L * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    initParticles<<<blocks, threads>>>(d_x, d_y, d_theta, d_states, N, 2.0f, 2.0f, 0.0f, 1.0f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_init = 0; cudaEventElapsedTime(&ms_init, start, stop);
    std::cout << "Init time (ms): " << ms_init << std::endl;

    for (int t = 0; t < 10; t++) {
        cudaEventRecord(start);
        propagateParticles<<<blocks, threads>>>(d_x, d_y, d_theta, 0.5f, 0.1f, d_states, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_prop; cudaEventElapsedTime(&ms_prop, start, stop);

        cudaEventRecord(start);
        computeWeights<<<blocks, threads>>>(d_x, d_y, d_weights, d_landmarks, d_z, N, L);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_update; cudaEventElapsedTime(&ms_update, start, stop);

        float* h_weights = new float[N];
        cudaMemcpy(h_weights, d_weights, N * sizeof(float), cudaMemcpyDeviceToHost);
        float sum = 0.0f; for (int i = 0; i < N; i++) sum += h_weights[i];
        delete[] h_weights;

        cudaEventRecord(start);
        normalizeWeights<<<blocks, threads>>>(d_weights, sum, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_norm; cudaEventElapsedTime(&ms_norm, start, stop);

        cudaEventRecord(start);
        resampleParticles<<<blocks, threads>>>(d_x, d_y, d_theta, d_x_new, d_y_new, d_theta_new, d_weights, d_states, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_resamp; cudaEventElapsedTime(&ms_resamp, start, stop);

        cudaMemcpy(d_x, d_x_new, N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_y, d_y_new, N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_theta, d_theta_new, N * sizeof(float), cudaMemcpyDeviceToDevice);

        printf("Iter %d — prop: %.3f ms, update: %.3f ms, norm: %.3f ms, resamp: %.3f ms\n", t, ms_prop, ms_update, ms_norm, ms_resamp);
    }

    std::cout << "Done." << std::endl;
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_theta);
    cudaFree(d_weights); cudaFree(d_x_new); cudaFree(d_y_new); cudaFree(d_theta_new);
    cudaFree(d_states); cudaFree(d_landmarks); cudaFree(d_z);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}