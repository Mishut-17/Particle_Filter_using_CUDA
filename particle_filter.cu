#include "particle_filter.cuh"
#include <cmath>
#include <curand_kernel.h>

__global__ void initParticles(float* x, float* y, float* theta, curandState* states, int N, float x0, float y0, float theta0, float spread) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        curand_init(1337ULL + i, 0, 0, &states[i]);
        x[i] = x0 + spread * (curand_uniform(&states[i]) - 0.5f);
        y[i] = y0 + spread * (curand_uniform(&states[i]) - 0.5f);
        theta[i] = theta0 + spread * (curand_uniform(&states[i]) - 0.5f);
    }
}
__global__ void propagateParticles(float* x, float* y, float* theta, float v, float w, curandState* states, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float nv = v + 0.1f * (curand_uniform(&states[i]) - 0.5f);
        float nw = w + 0.05f * (curand_uniform(&states[i]) - 0.5f);
        x[i] += nv * cosf(theta[i]);
        y[i] += nv * sinf(theta[i]);
        theta[i] += nw;
    }
}
__global__ void computeWeights(float* x, float* y, float* weights, Landmark* landmarks, float* z, int N, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float w_sum = 0.0f;
        for (int j = 0; j < L; j++) {
            float dx = x[i] - landmarks[j].x;
            float dy = y[i] - landmarks[j].y;
            float dist = sqrtf(dx * dx + dy * dy);
            float diff = dist - z[j];
            w_sum += expf(-0.5f * diff * diff / 0.25f);
        }
        weights[i] = w_sum;
    }
}
__global__ void normalizeWeights(float* weights, float sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) weights[i] /= sum;
}
__global__ void resampleParticles(float* x, float* y, float* theta, float* new_x, float* new_y, float* new_theta, float* weights, curandState* states, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float r = curand_uniform(&states[i]);
        float c = weights[0];
        int j = 0;
        while (r > c && j < N - 1) { j++; c += weights[j]; }
        new_x[i] = x[j]; new_y[i] = y[j]; new_theta[i] = theta[j];
    }
}