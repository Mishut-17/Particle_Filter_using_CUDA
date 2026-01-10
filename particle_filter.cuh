#ifndef PARTICLE_FILTER_CUH
#define PARTICLE_FILTER_CUH
#include <curand_kernel.h>
struct Landmark { float x, y; };
__global__ void initParticles(float* x, float* y, float* theta, curandState* states, int N, float x0, float y0, float theta0, float spread);
__global__ void propagateParticles(float* x, float* y, float* theta, float v, float w, curandState* states, int N);
__global__ void computeWeights(float* x, float* y, float* weights, Landmark* landmarks, float* z, int N, int L);
__global__ void normalizeWeights(float* weights, float sum, int N);
__global__ void resampleParticles(float* x, float* y, float* theta, float* new_x, float* new_y, float* new_theta, float* weights, curandState* states, int N);
#endif