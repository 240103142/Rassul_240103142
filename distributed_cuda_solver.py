#!/usr/bin/env python3
"""Distributed 2-D reaction-diffusion stencil for a 4-rank CUDA/MPI run.

Run on a Colab T4 (after installing mpi4py, numpy, and numba-cuda):

    mpirun --allow-run-as-root --oversubscribe -np 4 \
        python3 distributed_cuda_solver.py YOUR_GITHUB_USERNAME

This program writes submission_manifest.json on rank 0.  Hardware-dependent
values in that manifest are measured during the run; they must not be filled
with invented values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np
from mpi4py import MPI
from numba import cuda, float64


# ------------------------------ Lab constants ------------------------------
GLOBAL_NX = 512
GLOBAL_NY = 512
RANKS_X = 2
RANKS_Y = 2
LOCAL_NX = GLOBAL_NX // RANKS_X
LOCAL_NY = GLOBAL_NY // RANKS_Y
PAD_NX = LOCAL_NX + 2
PAD_NY = LOCAL_NY + 2
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2
STEPS = 400
LAMBDA_DAMP = 0.15
DX = 1.0 / (GLOBAL_NX + 1)
DY = 1.0 / (GLOBAL_NY + 1)
BLOCK = (TILE_DIM, TILE_DIM)

# The message tag names describe the direction in which a rank sends.
TAG_TOP = 100
TAG_BOTTOM = 101
TAG_LEFT = 102
TAG_RIGHT = 103


def get_gpu_uuid() -> str:
    """Return the active NVIDIA GPU UUID reported by nvidia-smi."""
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    )
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("nvidia-smi returned no GPU UUID.")
    return lines[0]


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive the required unsigned 64-bit seed from GitHub handle and UUID."""
    value = f"{github_user.strip()}::{gpu_uuid}"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def init_distributed_context(github_user: str):
    """Initialize MPI topology, select GPU 0, and derive the student seed."""
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    size = world.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks, received {size}.")
    if not github_user.strip() or github_user == "REPLACE_WITH_YOUR_GITHUB_USERNAME":
        raise ValueError("Pass your real GitHub username as the first argument.")

    cart = world.Create_cart(
        dims=(RANKS_Y, RANKS_X), periods=(False, False), reorder=False
    )
    cuda.select_device(0)

    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = world.bcast(gpu_uuid, root=0)
    seed = derive_student_seed(github_user, gpu_uuid)

    coords = cart.Get_coords(cart.Get_rank())
    nbr_up, nbr_down = cart.Shift(0, 1)
    nbr_left, nbr_right = cart.Shift(1, 1)
    neighbors = (nbr_up, nbr_down, nbr_left, nbr_right)
    return world, cart, rank, tuple(coords), neighbors, gpu_uuid, seed


def initialize_subdomains(coords, neighbors, seed: int):
    """Create only this rank's padded 258x258 arrays on the host."""
    row_rank, col_rank = coords

    global_rows = row_rank * LOCAL_NY + (np.arange(PAD_NY) - 1)
    global_cols = col_rank * LOCAL_NX + (np.arange(PAD_NX) - 1)
    ys = (global_rows + 1.0) * DY
    xs = (global_cols + 1.0) * DX
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    # Smooth heterogeneous diffusivity with a guaranteed [0.08, 0.24] range.
    kappa = 0.16 + 0.08 * np.sin(2.0 * math.pi * xx) * np.cos(2.0 * math.pi * yy)
    kappa = np.clip(kappa, 0.08, 0.24).astype(np.float64, copy=False)

    rng = np.random.default_rng(seed)
    pulse_specs = []
    for _ in range(3):
        cx, cy = rng.uniform(0.20, 0.80, size=2)
        amplitude = rng.uniform(0.55, 0.95)
        sigma = rng.uniform(0.035, 0.065)
        pulse_specs.append((float(cx), float(cy), float(amplitude), float(sigma)))

    u = np.zeros((PAD_NY, PAD_NX), dtype=np.float64)
    for cx, cy, amplitude, sigma in pulse_specs:
        radius2 = (xx - cx) ** 2 + (yy - cy) ** 2
        u += amplitude * np.exp(-radius2 / (2.0 * sigma * sigma))

    # A weak, localized non-positive source acts as a sink so the Q6 energy
    # trace is dissipative while still exercising the spatial source field.
    sx, sy = pulse_specs[0][0], pulse_specs[0][1]
    source_radius2 = (xx - sx) ** 2 + (yy - sy) ** 2
    source = -1.0e-3 * np.exp(-source_radius2 / (2.0 * 0.045 * 0.045))
    source = source.astype(np.float64, copy=False)

    nbr_up, nbr_down, nbr_left, nbr_right = neighbors
    if nbr_up == MPI.PROC_NULL:
        u[0, :] = 0.0
    if nbr_down == MPI.PROC_NULL:
        u[LOCAL_NY + 1, :] = 0.0
    if nbr_left == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if nbr_right == MPI.PROC_NULL:
        u[:, LOCAL_NX + 1] = 0.0

    return u, kappa, source, pulse_specs


# --------------------------- GPU halo primitives ---------------------------
@cuda.jit
def pack_halos_kernel(u, send_top, send_bottom, send_left, send_right, nx, ny):
    index = cuda.grid(1)
    if index < nx:
        send_top[index] = u[1, index + 1]
        send_bottom[index] = u[ny, index + 1]
    if index < ny:
        send_left[index] = u[index + 1, 1]
        send_right[index] = u[index + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, recv_top, recv_bottom, recv_left, recv_right, nx, ny):
    index = cuda.grid(1)
    if index < nx:
        u[0, index + 1] = recv_top[index]
        u[ny + 1, index + 1] = recv_bottom[index]
    if index < ny:
        u[index + 1, 0] = recv_left[index]
        u[index + 1, nx + 1] = recv_right[index]


@dataclass
class HaloBuffers:
    send_host: tuple
    recv_host: tuple
    send_device: tuple
    recv_device: tuple


def make_halo_buffers():
    """Allocate persistent pinned host and contiguous device halo vectors."""
    send_host = tuple(cuda.pinned_array(LOCAL_NX, dtype=np.float64) for _ in range(4))
    recv_host = tuple(cuda.pinned_array(LOCAL_NX, dtype=np.float64) for _ in range(4))
    send_device = tuple(cuda.device_array(LOCAL_NX, dtype=np.float64) for _ in range(4))
    recv_device = tuple(cuda.device_array(LOCAL_NX, dtype=np.float64) for _ in range(4))
    return HaloBuffers(send_host, recv_host, send_device, recv_device)


def exchange_halos_async(u, cart, neighbors, buffers, stream_boundary):
    """Pack on GPU, stage D2H, then post tag-matched nonblocking MPI calls."""
    pack_halos_kernel[(LOCAL_NX + 127) // 128, 128, stream_boundary](
        u,
        buffers.send_device[0],
        buffers.send_device[1],
        buffers.send_device[2],
        buffers.send_device[3],
        LOCAL_NX,
        LOCAL_NY,
    )
    for device_vec, host_vec in zip(buffers.send_device, buffers.send_host):
        device_vec.copy_to_host(host_vec, stream=stream_boundary)
    # Host MPI must not read a pinned send vector before its asynchronous D2H
    # transfer completes.
    stream_boundary.synchronize()

    # MPI leaves PROC_NULL receives untouched; clear all receive vectors first
    # so unpacking preserves homogeneous zero Dirichlet physical boundaries.
    for host_vec in buffers.recv_host:
        host_vec.fill(0.0)

    nbr_up, nbr_down, nbr_left, nbr_right = neighbors
    requests = [
        cart.Irecv(buffers.recv_host[0], source=nbr_up, tag=TAG_BOTTOM),
        cart.Irecv(buffers.recv_host[1], source=nbr_down, tag=TAG_TOP),
        cart.Irecv(buffers.recv_host[2], source=nbr_left, tag=TAG_RIGHT),
        cart.Irecv(buffers.recv_host[3], source=nbr_right, tag=TAG_LEFT),
        cart.Isend(buffers.send_host[0], dest=nbr_up, tag=TAG_TOP),
        cart.Isend(buffers.send_host[1], dest=nbr_down, tag=TAG_BOTTOM),
        cart.Isend(buffers.send_host[2], dest=nbr_left, tag=TAG_LEFT),
        cart.Isend(buffers.send_host[3], dest=nbr_right, tag=TAG_RIGHT),
    ]
    return requests


def complete_halos_async(u, cart, requests, buffers, stream_boundary):
    """Wait for MPI, stage H2D on stream_boundary, and unpack ghost cells."""
    MPI.Request.Waitall(requests)
    for host_vec, device_vec in zip(buffers.recv_host, buffers.recv_device):
        device_vec.copy_to_device(host_vec, stream=stream_boundary)
    unpack_halos_kernel[(LOCAL_NX + 127) // 128, 128, stream_boundary](
        u,
        buffers.recv_device[0],
        buffers.recv_device[1],
        buffers.recv_device[2],
        buffers.recv_device[3],
        LOCAL_NX,
        LOCAL_NY,
    )


# -------------------------- Shared-memory stencil --------------------------
@cuda.jit
def tiled_stencil_kernel(
    u,
    u_next,
    kappa,
    source,
    nx,
    ny,
    r_start,
    r_end,
    c_start,
    c_end,
    ring_only,
    dt,
    dx,
    dy,
    damping,
):
    """Update a box using a 16x16 block and an 18x18 float64 shared tile."""
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = r_start + cuda.blockIdx.y * TILE_DIM + ty
    col = c_start + cuda.blockIdx.x * TILE_DIM + tx
    tile = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)

    # Every thread stages its center cell. Guard the padded 258x258 array for
    # inactive threads at the edges of a partial bounding-box tile.
    if 0 <= row < ny + 2 and 0 <= col < nx + 2:
        tile[ty + 1, tx + 1] = u[row, col]
    else:
        tile[ty + 1, tx + 1] = 0.0

    # The four one-cell apron strips are loaded by the edge threads. The
    # 5-point stencil does not read the four apron corners.
    if tx == 0:
        if 0 <= row < ny + 2 and 0 <= col - 1 < nx + 2:
            tile[ty + 1, 0] = u[row, col - 1]
        else:
            tile[ty + 1, 0] = 0.0
    if tx == TILE_DIM - 1:
        if 0 <= row < ny + 2 and 0 <= col + 1 < nx + 2:
            tile[ty + 1, TILE_DIM + 1] = u[row, col + 1]
        else:
            tile[ty + 1, TILE_DIM + 1] = 0.0
    if ty == 0:
        if 0 <= row - 1 < ny + 2 and 0 <= col < nx + 2:
            tile[0, tx + 1] = u[row - 1, col]
        else:
            tile[0, tx + 1] = 0.0
    if ty == TILE_DIM - 1:
        if 0 <= row + 1 < ny + 2 and 0 <= col < nx + 2:
            tile[TILE_DIM + 1, tx + 1] = u[row + 1, col]
        else:
            tile[TILE_DIM + 1, tx + 1] = 0.0

    # Required block-wide barrier: apron values can be produced by another warp.
    cuda.syncthreads()

    if row < r_start or row >= r_end or col < c_start or col >= c_end:
        return
    if row < 1 or row > ny or col < 1 or col > nx:
        return
    if ring_only != 0 and not (
        row == 1 or row == ny or col == 1 or col == nx
    ):
        return

    center = tile[ty + 1, tx + 1]
    north = tile[ty, tx + 1]
    south = tile[ty + 2, tx + 1]
    west = tile[ty + 1, tx]
    east = tile[ty + 1, tx + 2]

    k_center = kappa[row, col]
    k_north = 0.5 * (k_center + kappa[row - 1, col])
    k_south = 0.5 * (k_center + kappa[row + 1, col])
    k_west = 0.5 * (k_center + kappa[row, col - 1])
    k_east = 0.5 * (k_center + kappa[row, col + 1])

    flux_y = (k_south * (south - center) - k_north * (center - north)) / (dy * dy)
    flux_x = (k_east * (east - center) - k_west * (center - west)) / (dx * dx)
    reaction = -damping * center * center * center
    u_next[row, col] = center + dt * (flux_y + flux_x + reaction + source[row, col])


def launch_stencil(
    u,
    u_next,
    kappa,
    source,
    bounds,
    ring_only,
    dt,
    stream,
):
    r_start, r_end, c_start, c_end = bounds
    grid = (
        (c_end - c_start + TILE_DIM - 1) // TILE_DIM,
        (r_end - r_start + TILE_DIM - 1) // TILE_DIM,
    )
    tiled_stencil_kernel[grid, BLOCK, stream](
        u,
        u_next,
        kappa,
        source,
        LOCAL_NX,
        LOCAL_NY,
        r_start,
        r_end,
        c_start,
        c_end,
        ring_only,
        dt,
        DX,
        DY,
        LAMBDA_DAMP,
    )


def inspect_ptx() -> dict[str, int]:
    """Count required shared-memory and block-barrier PTX instructions."""
    assembly = tiled_stencil_kernel.inspect_asm()
    if isinstance(assembly, dict):
        ptx = "\n".join(str(value) for value in assembly.values())
    else:
        ptx = str(assembly)
    lines = ptx.splitlines()
    counts = {
        "st.shared.f64": sum(bool(re.match(r"^\s*st\.shared\.f64\b", line)) for line in lines),
        "ld.shared.f64": sum(bool(re.match(r"^\s*ld\.shared\.f64\b", line)) for line in lines),
        "bar.sync": sum(bool(re.match(r"^\s*bar\.sync\b", line)) for line in lines),
    }
    if any(counts[name] == 0 for name in counts):
        raise RuntimeError(f"PTX audit failed: expected nonzero counts, got {counts}")
    return counts


# -------------------------- Time-stepping control --------------------------
@dataclass
class SolverState:
    u: Any
    u_next: Any
    kappa: Any
    source: Any
    buffers: HaloBuffers
    stream_boundary: Any
    stream_interior: Any


def make_solver_state(u_host, kappa_host, source_host) -> SolverState:
    return SolverState(
        u=cuda.to_device(u_host),
        u_next=cuda.device_array((PAD_NY, PAD_NX), dtype=np.float64),
        kappa=cuda.to_device(kappa_host),
        source=cuda.to_device(source_host),
        buffers=make_halo_buffers(),
        stream_boundary=cuda.stream(),
        stream_interior=cuda.stream(),
    )


def compute_checkpoint(state, cart, step: int):
    local = state.u.copy_to_host()
    owned = local[1 : LOCAL_NY + 1, 1 : LOCAL_NX + 1]
    local_energy = float(np.sum(owned, dtype=np.float64) * DX * DY)
    if step == 0:
        local_residual_sq = 0.0
    else:
        # After the ping-pong swap, u_next retains the immediately previous
        # timestep for every owned cell. This is a one-step L2 update norm.
        previous_local = state.u_next.copy_to_host()
        previous_owned = previous_local[1 : LOCAL_NY + 1, 1 : LOCAL_NX + 1]
        diff = owned - previous_owned
        local_residual_sq = float(np.sum(diff * diff, dtype=np.float64))

    send = np.array([local_energy, local_residual_sq], dtype=np.float64)
    recv = np.empty_like(send)
    cart.Allreduce(send, recv, op=MPI.SUM)
    energy = float(recv[0])
    residual = float(math.sqrt(max(0.0, recv[1])))
    return {"step": step, "energy": energy, "l2_residual": residual}


def run_solver(
    mode: str,
    u_initial,
    kappa_host,
    source_host,
    cart,
    neighbors,
    dt: float,
    steps: int,
    rank: int,
):
    """Run all steps in serialized or overlapped mode and return local result."""
    if mode not in ("serialized", "overlapped"):
        raise ValueError("mode must be 'serialized' or 'overlapped'")
    state = make_solver_state(u_initial, kappa_host, source_host)
    stream_boundary = state.stream_boundary
    stream_interior = state.stream_interior

    # Warm up/compile kernels before wall-clock measurement so setup time does
    # not pollute the 400-step comparison. The first full step overwrites this
    # scratch target before it is read.
    warm_grid = ((LOCAL_NX + TILE_DIM - 1) // TILE_DIM,
                 (LOCAL_NY + TILE_DIM - 1) // TILE_DIM)
    tiled_stencil_kernel[warm_grid, BLOCK, stream_boundary](
        state.u, state.u_next, state.kappa, state.source,
        LOCAL_NX, LOCAL_NY, 1, LOCAL_NY + 1, 1, LOCAL_NX + 1,
        0, dt, DX, DY, LAMBDA_DAMP,
    )
    stream_boundary.synchronize()

    trace = []
    row = compute_checkpoint(state, cart, 0)
    trace.append(row)
    cuda.synchronize()
    cart.Barrier()

    boundary_start = cuda.event(timing=True)
    boundary_end = cuda.event(timing=True)
    boundary_start.record(stream_boundary)
    if mode == "overlapped":
        interior_start = cuda.event(timing=True)
        interior_end = cuda.event(timing=True)
        interior_start.record(stream_interior)
    else:
        interior_start = interior_end = None

    started = MPI.Wtime()
    for step in range(1, steps + 1):
        requests = exchange_halos_async(
            state.u, cart, neighbors, state.buffers, stream_boundary
        )

        if mode == "overlapped":
            # Deep cells [2..ny-1, 2..nx-1] do not depend on fresh halos.
            launch_stencil(
                state.u, state.u_next, state.kappa, state.source,
                (2, LOCAL_NY, 2, LOCAL_NX), 0, dt, stream_interior,
            )

        complete_halos_async(
            state.u, cart, requests, state.buffers, stream_boundary
        )

        if mode == "serialized":
            launch_stencil(
                state.u, state.u_next, state.kappa, state.source,
                (1, LOCAL_NY + 1, 1, LOCAL_NX + 1), 0, dt, stream_boundary,
            )
            stream_boundary.synchronize()
        else:
            # Outer one-cell ring depends on the just-unpacked ghost halos.
            launch_stencil(
                state.u, state.u_next, state.kappa, state.source,
                (1, LOCAL_NY + 1, 1, LOCAL_NX + 1), 1, dt, stream_boundary,
            )
            stream_boundary.synchronize()
            stream_interior.synchronize()

        state.u, state.u_next = state.u_next, state.u

        if step % 100 == 0 or step == steps:
            row = compute_checkpoint(state, cart, step)
            trace.append(row)
            if rank == 0:
                print(
                    f"{mode} step={step:3d} energy={row['energy']:.16e} "
                    f"l2_residual={row['l2_residual']:.16e}",
                    flush=True,
                )

    cuda.synchronize()
    elapsed_local = MPI.Wtime() - started
    cart.Barrier()
    wall_ms = float(cart.allreduce(elapsed_local, op=MPI.MAX) * 1000.0)

    boundary_end.record(stream_boundary)
    boundary_end.synchronize()
    boundary_event_ms = float(cuda.event_elapsed_time(boundary_start, boundary_end))
    if mode == "overlapped":
        interior_end.record(stream_interior)
        interior_end.synchronize()
        interior_event_ms = float(cuda.event_elapsed_time(interior_start, interior_end))
    else:
        interior_event_ms = 0.0

    final_local = state.u.copy_to_host()
    result = {
        "mode": mode,
        "wall_ms_max_rank": wall_ms,
        "boundary_stream_event_ms": boundary_event_ms,
        "interior_stream_event_ms": interior_event_ms,
        "trace": trace if rank == 0 else None,
        "final_local": final_local,
    }
    return result


def run_ptx_warmup(u_initial, kappa_host, source_host, dt: float):
    """Compile the required tiled stencil once so inspect_asm has a signature."""
    d_u = cuda.to_device(u_initial)
    d_next = cuda.device_array((PAD_NY, PAD_NX), dtype=np.float64)
    d_kappa = cuda.to_device(kappa_host)
    d_source = cuda.to_device(source_host)
    grid = ((LOCAL_NX + TILE_DIM - 1) // TILE_DIM,
            (LOCAL_NY + TILE_DIM - 1) // TILE_DIM)
    tiled_stencil_kernel[grid, BLOCK](
        d_u, d_next, d_kappa, d_source,
        LOCAL_NX, LOCAL_NY, 1, LOCAL_NY + 1, 1, LOCAL_NX + 1,
        0, dt, DX, DY, LAMBDA_DAMP,
    )
    cuda.synchronize()
    return inspect_ptx()


def run_main(github_user: str, manifest_path: str, steps: int):
    world, cart, rank, coords, neighbors, gpu_uuid, seed = init_distributed_context(
        github_user
    )
    u_initial, kappa_host, source_host, pulse_specs = initialize_subdomains(
        coords, neighbors, seed
    )

    local_kappa_max = float(np.max(kappa_host[1 : LOCAL_NY + 1, 1 : LOCAL_NX + 1]))
    kappa_max = float(cart.allreduce(local_kappa_max, op=MPI.MAX))
    dt_max = DX * DX / (4.0 * kappa_max)
    dt = 0.8 * dt_max

    if rank == 0:
        print(f"GITHUB_USERNAME={github_user.strip()}")
        print(f"GPU_UUID={gpu_uuid}")
        print(f"INSTANCE_SEED={seed}")
        print(f"MPI ranks={cart.Get_size()} cartesian_dims={RANKS_Y}x{RANKS_X}")
        print(f"local_shape=({PAD_NY}, {PAD_NX}) dt={dt:.16e}")
        print(f"kappa_max={kappa_max:.16e} dt_max={dt_max:.16e}")

    ptx_counts = run_ptx_warmup(u_initial, kappa_host, source_host, dt)
    if rank == 0:
        print("PTX_AUDIT=" + json.dumps(ptx_counts, sort_keys=True))

    serial = run_solver(
        "serialized", u_initial, kappa_host, source_host, cart, neighbors,
        dt, steps, rank,
    )
    overlap = run_solver(
        "overlapped", u_initial, kappa_host, source_host, cart, neighbors,
        dt, steps, rank,
    )

    serial_owned = serial["final_local"][1 : LOCAL_NY + 1, 1 : LOCAL_NX + 1]
    overlap_owned = overlap["final_local"][1 : LOCAL_NY + 1, 1 : LOCAL_NX + 1]
    local_max_diff = float(np.max(np.abs(serial_owned - overlap_owned)))
    parity_max_diff = float(cart.allreduce(local_max_diff, op=MPI.MAX))
    parity_passed = parity_max_diff < 1.0e-12
    if rank == 0:
        print(f"parity_max_diff={parity_max_diff:.16e} parity_passed={parity_passed}")
        print(f"serialized_ms={serial['wall_ms_max_rank']:.6f}")
        print(f"overlapped_ms={overlap['wall_ms_max_rank']:.6f}")

    manifest = {
        "github_username": github_user.strip(),
        "gpu_uuid": gpu_uuid,
        "instance_seed": seed,
        "mpi_ranks": cart.Get_size(),
        "cartesian_dims": [RANKS_Y, RANKS_X],
        "periods": [False, False],
        "global_shape": [GLOBAL_NY, GLOBAL_NX],
        "local_shape_with_ghosts": [PAD_NY, PAD_NX],
        "steps": steps,
        "dx": DX,
        "dy": DY,
        "dt": dt,
        "kappa_max": kappa_max,
        "dt_max_diffusion": dt_max,
        "lambda_damp": LAMBDA_DAMP,
        "source_is_nonpositive_localized_sink": True,
        "pulse_specs": pulse_specs,
        "ptx_audit": ptx_counts,
        "ptx_audit_passed": all(value > 0 for value in ptx_counts.values()),
        "parity_max_diff": parity_max_diff,
        "parity_passed": parity_passed,
        "serialized_ms": serial["wall_ms_max_rank"],
        "overlapped_ms": overlap["wall_ms_max_rank"],
        "cuda_event_ms": {
            "serialized_boundary_stream": serial["boundary_stream_event_ms"],
            "overlapped_boundary_stream": overlap["boundary_stream_event_ms"],
            "overlapped_interior_stream": overlap["interior_stream_event_ms"],
        },
        "energy_trace": {
            "serialized": serial["trace"],
            "overlapped": overlap["trace"],
        },
    }

    if rank == 0:
        with open(manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        print(f"Wrote {manifest_path}")
    world.Barrier()
    if not parity_passed:
        raise RuntimeError(
            f"Serialized/overlapped parity failed: max abs diff={parity_max_diff:.3e}"
        )
    return manifest if rank == 0 else None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("github_username", help="Exact GitHub account handle")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--manifest", default="submission_manifest.json")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_main(args.github_username, args.manifest, args.steps)


if __name__ == "__main__":
    main()
