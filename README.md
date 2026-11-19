# Distributed Reaction-Diffusion Solver with MPI and CUDA

## Project Overview

This project implements a two-dimensional reaction-diffusion solver using four
MPI processes and a CUDA GPU. The 512 × 512 interior domain is divided across a
non-periodic 2 × 2 Cartesian process grid. Each process owns a 256 × 256
subdomain and stores it with a one-cell ghost-cell halo, giving a local array
size of 258 × 258.

## Numerical Model

The solver uses explicit time stepping and a flux-form five-point stencil with
a spatially varying thermal diffusivity in the range [0.08, 0.24]. The reaction
term uses a cubic damping coefficient of 0.15. The initial field consists of
seeded Gaussian thermal pulses, and the source field is a weak localized sink
to support a dissipative energy trace. Homogeneous Dirichlet conditions are
applied at the physical boundaries. The time step is set below the diffusion
CFL stability limit.

## Parallel Implementation

- Each MPI process stores only its local fields; global field gathering is not
  used.
- Dedicated CUDA kernels pack and unpack the four halo faces. Pinned host
  buffers stage asynchronous device-to-host and host-to-device transfers for
  nonblocking MPI communication.
- The stencil uses 16 × 16 thread blocks and an 18 × 18 float64 shared-memory
  tile. A block-wide barrier separates tile and halo loading from stencil
  evaluation.
- The overlapped pipeline computes the halo-independent interior on a separate
  CUDA stream while MPI transfers are in progress, then updates the perimeter
  after the received halos are unpacked.
- Serialized and overlapped modes each run for 400 steps. Every 100 steps, the
  program computes global thermal energy and the global L2 norm of the latest
  one-step update using MPI reductions.
- The generated PTX is inspected for shared-memory load/store instructions and
  block barriers. The two final local states are compared against a maximum
  absolute-difference threshold of 1e-12.

## Source and Runtime Results

The solver is provided in [distributed_cuda_solver.py](distributed_cuda_solver.py).
Its runtime identity seed is derived from the GitHub username and active GPU
UUID. After a successful run, the program writes `submission_manifest.json`
with the seed, GPU identifier, PTX audit counts, measured timings, energy
trace, and serialized-versus-overlapped parity result. GPU-dependent values
are recorded from the actual Tesla T4 execution environment.
