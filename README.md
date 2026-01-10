#BOX BLUR

🚀 CUDA Particle Filter – GPU Capstone Project
📌 Overview

This project implements a CUDA-accelerated Particle Filter for real-time state estimation and tracking.
The goal is to demonstrate how GPU parallelism can drastically speed up computationally expensive algorithms by processing thousands of particles simultaneously.

All major stages of the particle filter are executed on the GPU:

Initialization

Propagation

Weight update

Normalization

Resampling

This project was developed as part of the GPU Specialization Capstone.

🧠 Algorithm Summary

A particle filter estimates the state of a system using a large set of random samples (particles).
Each iteration performs:

Propagation
Updates each particle using a motion model.

Update
Computes the likelihood of each particle based on sensor measurements.

Normalization
Normalizes all particle weights so they sum to 1.

Resampling
Generates a new particle set based on weight distribution.

GPU parallelism assigns one thread per particle, enabling massive speedup over CPU implementations.

📂 Project Structure
.
├── main.cu                # Program entry point
├── particle_filter.cu     # CUDA kernel implementations
├── particle_filter.cuh    # Structs and kernel declarations
├── pgm_images.zip         # Test PGM images (synthetic data)
└── README.md

⚙️ Requirements

NVIDIA GPU

CUDA Toolkit (12.4+ recommended)

Linux / Google Colab / WSL (recommended)

nvcc compiler

🔧 Compilation

Navigate to project folder and run:

nvcc -o particle_filter main.cu particle_filter.cu

▶️ Execution

Run:

./particle_filter


If your program takes arguments:

./particle_filter input_file particle_count

📊 Sample Output
Init time (ms): 11.3092
Iter 0 — prop: 0.007 ms, update: 0.003 ms, norm: 0.003 ms, resamp: 0.003 ms
Iter 1 — prop: 0.003 ms, update: 0.004 ms, norm: 0.003 ms, resamp: 0.003 ms
...
Iter 9 — prop: 0.004 ms, update: 0.003 ms, norm: 0.003 ms, resamp: 0.003 ms
Done.

Interpretation

Init time → GPU memory setup and particle initialization

prop → propagation kernel

update → weight update kernel

norm → normalization kernel

resamp → resampling kernel

Each iteration completes in ~0.012 ms, proving GPU acceleration efficiency.

🖼 Test Data

The repository includes 10 synthetic PGM images used for testing:

256×256 grayscale images

Stored in pgm_images.zip

Useful for benchmarking and validation

📈 Performance Highlights

Massive parallelism: one thread per particle

Sub-millisecond kernel execution

Scales efficiently with particle count

Minimal host-device memory transfers

🧪 What I Learned

CUDA kernel design

GPU memory management (cudaMalloc, cudaMemcpy)

Kernel launch configuration

Parallel reduction techniques

Performance profiling

GPU optimization strategies

🔮 Future Improvements

Support for real sensor / camera input

Shared memory optimization

Thrust prefix-sum based resampling

Visual particle animation

Multi-GPU scaling

🎯 Conclusion

This project demonstrates how GPU computing transforms particle filtering from a slow CPU process into a real-time parallel system.
It validates CUDA’s power for scientific computing and robotics applications.

👤 Author

Utkarsh Mishra
GPU Specialization Capstone Project
