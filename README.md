🚀 CUDA Particle Filter – GPU Capstone Project
📌 Overview

This project implements a CUDA-accelerated Particle Filter for real-time state estimation and tracking.
The objective is to demonstrate how GPU parallelism significantly improves performance by processing thousands of particles simultaneously.

All major stages of the particle filter run on the GPU:

Initialization

Propagation

Weight update

Normalization

Resampling

This project was developed as part of the GPU Specialization Capstone.

🧠 Algorithm Summary

A particle filter estimates the state of a system using many random samples called particles.

Each iteration performs:

Propagation
Updates each particle using a motion model.

Update
Computes how well each particle matches the measurement.

Normalization
Scales all weights so their sum equals 1.

Resampling
Generates a new particle set based on weight distribution.

GPU parallelism assigns one thread per particle, enabling massive speedup over CPU implementations.

📂 Project Structure
.
├── main.cu
├── particle_filter.cu
├── particle_filter.cuh
├── pgm_images.zip
└── README.md

⚙️ Requirements

NVIDIA GPU

CUDA Toolkit (12.4+ recommended)

Linux / WSL / Google Colab

nvcc compiler

🔧 Compilation
nvcc -o particle_filter main.cu particle_filter.cu

▶️ Execution
./particle_filter


If arguments are required:

./particle_filter input_file particle_count

📊 Sample Output
Init time (ms): 11.3092
Iter 0 — prop: 0.007 ms, update: 0.003 ms, norm: 0.003 ms, resamp: 0.003 ms
Iter 1 — prop: 0.003 ms, update: 0.004 ms, norm: 0.003 ms, resamp: 0.003 ms
Iter 2 — prop: 0.004 ms, update: 0.003 ms, norm: 0.003 ms, resamp: 0.003 ms
...
Iter 9 — prop: 0.004 ms, update: 0.003 ms, norm: 0.003 ms, resamp: 0.003 ms
Done.

Output Explanation

Init time → GPU memory setup & initialization

prop → particle propagation

update → weight computation

norm → weight normalization

resamp → particle resampling

Each iteration completes in ~0.012 ms, proving efficient GPU acceleration.

🖼 Test Data

The repository includes 10 synthetic PGM images:

256×256 grayscale images

Stored in pgm_images.zip

Used for benchmarking and testing

📈 Performance Highlights

One GPU thread per particle

Sub-millisecond kernel execution

Efficient scaling with particle count

Minimal host-device transfers

🧪 What I Learned

CUDA kernel design

GPU memory management

Parallel reduction

Performance profiling

Kernel optimization strategies

🔮 Future Work

Real sensor / camera input

Shared memory optimization

Thrust-based prefix sum resampling

Visualization of particle movement

Multi-GPU scaling

🎯 Conclusion

This project demonstrates how GPU computing transforms particle filtering from a slow CPU process into a real-time parallel system.
It highlights CUDA’s power for scientific and robotics applications.

👤 Author

Utkarsh Mishra
GPU Specialization Capstone Project
