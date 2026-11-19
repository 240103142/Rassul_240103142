# Parallel and Multi-Core Systems Laboratories

This repository contains two Python laboratory projects on distributed
computing and CPU cache-coherence behavior.

## 1. Distributed Reaction-Diffusion Solver with MPI and CUDA

The first project implements a two-dimensional reaction-diffusion solver using
four MPI processes and a CUDA GPU. A 512 × 512 interior domain is divided over
a non-periodic 2 × 2 Cartesian process grid. Each process owns a 256 × 256
subdomain with a one-cell ghost halo, giving a local 258 × 258 array.

The implementation uses GPU halo pack/unpack kernels, pinned host staging,
nonblocking MPI exchange, a 16 × 16 shared-memory stencil tile, and separate
CUDA streams for interior computation and perimeter updates. It compares
serialized and overlapped execution, performs MPI reductions, audits generated
PTX, and writes a runtime manifest after execution on the target GPU.

Source file: [distributed_cuda_solver.py](distributed_cuda_solver.py)

## 2. Cache-Line and False-Sharing Benchmark

The second project uses four Numba-compiled native threads to compare counter
updates at element strides 1, 4, 8, and 16 with a register-private baseline.
Each worker performs 60,000,000 iterations per trial. The benchmark records
three trials per layout, context-switch counts, runtime CPU topology, exact
counter parity, and LLVM IR load/store instructions.

The recorded host is an Apple M1 with a detected 128-byte cache line. The
median stride-1 time was 107,844,500 ns and the median stride-8 time was
39,625,583 ns, a 2.721588× slowdown. Because the detected line is 128 bytes,
stride 8 still places two workers in each line; stride 16 gives each worker a
separate line. The benchmark verification passed.

Files:

- [false_sharing_lab.py](false_sharing_lab.py)
- [cache_lab_manifest.json](cache_lab_manifest.json)
- [ANALYSIS.md](ANALYSIS.md)
