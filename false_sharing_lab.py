#!/usr/bin/env python3
"""Cache-line and false-sharing benchmark required by the laboratory brief."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import sysconfig
import threading
import time
from typing import Any

import numpy as np
import numba
from llvmlite import ir
from llvmlite.ir.instructions import LoadInstr, StoreInstr
from numba import njit, types
from numba.extending import intrinsic

try:
    import resource
except ImportError:
    resource = None


GIT_USERNAME = "240103142"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)
BENCHMARK_TRIALS = 3
UINT64_MASK = (1 << 64) - 1


def _enable_volatile_ir_rendering():
    """Teach this llvmlite version's IR printer to retain volatile operations."""
    if getattr(LoadInstr, "_cache_lab_volatile_patch", False):
        return

    original_load_descr = LoadInstr.descr
    original_store_descr = StoreInstr.descr

    def load_descr(instruction, buffer):
        if not getattr(instruction, "_cache_lab_volatile", False):
            return original_load_descr(instruction, buffer)
        (pointer,) = instruction.operands
        alignment = (
            f", align {instruction.align}" if instruction.align is not None else ""
        )
        metadata = instruction._stringify_metadata(leading_comma=True)
        buffer.append(
            f"load volatile {instruction.type}, {pointer.type} "
            f"{pointer.get_reference()}{alignment}{metadata}\n"
        )

    def store_descr(instruction, buffer):
        if not getattr(instruction, "_cache_lab_volatile", False):
            return original_store_descr(instruction, buffer)
        value, pointer = instruction.operands
        alignment = (
            f", align {instruction.align}" if instruction.align is not None else ""
        )
        metadata = instruction._stringify_metadata(leading_comma=True)
        buffer.append(
            f"store volatile {value.type} {value.get_reference()}, "
            f"{pointer.type} {pointer.get_reference()}{alignment}{metadata}\n"
        )

    LoadInstr.descr = load_descr
    StoreInstr.descr = store_descr
    LoadInstr._cache_lab_volatile_patch = True


_enable_volatile_ir_rendering()


def detect_cache_line_size() -> int:
    """Read the L1 data-cache coherency line size, with OS fallbacks."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        try:
            with open(linux_path, "r", encoding="utf-8") as stream:
                size = int(stream.read().strip())
            if size > 0:
                return size
        except (OSError, ValueError):
            pass
    if sys.platform == "darwin":
        try:
            value = subprocess.check_output(
                ["sysctl", "-n", "hw.cachelinesize"], text=True
            ).strip()
            size = int(value)
            if size > 0:
                return size
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return 64


def detect_cpu_model() -> str:
    """Return a model string suitable for the runtime identity seed."""
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                value = subprocess.check_output(
                    ["sysctl", "-n", key], text=True
                ).strip()
                if value:
                    return value
            except (OSError, subprocess.SubprocessError):
                pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as stream:
                for line in stream:
                    if line.lower().startswith(("model name", "hardware", "cpu model")):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor().strip() or platform.machine() or "unknown-cpu"


def init_system_context(git_user: str) -> dict[str, Any]:
    """Collect CPU/cache topology, affinity support, and deterministic seed."""
    git_user = git_user.strip()
    if not git_user or git_user == "REPLACE_WITH_YOUR_GIT_USERNAME":
        raise ValueError("GIT_USERNAME must be the real Git account handle.")

    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(
            f"Requires at least {NUM_THREADS} logical cores; found {logical_cores}."
        )

    line_size = detect_cache_line_size()
    cpu_model = detect_cpu_model()
    seed_input = f"{git_user}::{cpu_model}"
    seed = int(hashlib.sha256(seed_input.encode("utf-8")).hexdigest()[:16], 16)

    affinity_supported = hasattr(os, "sched_setaffinity")
    available_cpus = None
    assigned_cpus = None
    if affinity_supported and hasattr(os, "sched_getaffinity"):
        try:
            available_cpus = sorted(os.sched_getaffinity(0))
            if len(available_cpus) >= NUM_THREADS:
                assigned_cpus = available_cpus[:NUM_THREADS]
            else:
                affinity_supported = False
        except OSError:
            affinity_supported = False

    gil_enabled = (
        bool(sys._is_gil_enabled())
        if hasattr(sys, "_is_gil_enabled")
        else not bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    )

    return {
        "git_username": git_user,
        "instance_seed": seed,
        "cpu_model": cpu_model,
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "gil_enabled": gil_enabled,
        "logical_cores": logical_cores,
        "cache_line_size_bytes": line_size,
        "affinity_supported": bool(affinity_supported),
        "available_cpu_ids": available_cpus,
        "assigned_cpu_ids": assigned_cpus,
        "numba_version": numba.__version__,
        "numpy_version": np.__version__,
    }


@intrinsic
def _volatile_increment(typing_context, array, slot):
    """Emit one volatile uint64 load/add/store, preventing loop promotion."""
    if not isinstance(array, types.Array):
        return None
    if array.dtype != types.uint64 or array.ndim != 1 or array.layout != "C":
        return None
    if not isinstance(slot, types.Integer):
        return None

    signature = types.uint64(array, slot)

    def codegen(context, builder, sig, args):
        native_array = context.make_array(sig.args[0])(context, builder, args[0])
        index = context.cast(builder, args[1], sig.args[1], types.intp)
        element_type = context.get_value_type(types.uint64)
        address = builder.gep(
            native_array.data,
            [index],
            source_etype=element_type,
            inbounds=True,
        )
        old_value = builder.load(address, typ=element_type, align=8)
        old_value._cache_lab_volatile = True
        new_value = builder.add(old_value, ir.Constant(element_type, 1))
        store = builder.store(new_value, address, align=8)
        store._cache_lab_volatile = True
        return new_value

    return signature, codegen


@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, thread_seed):
    """Update the assigned memory slot on every iteration using volatile RMW."""
    if buffer[slot] != thread_seed:
        return buffer[slot]
    last_value = thread_seed
    for _ in range(iters):
        last_value = _volatile_increment(buffer, slot)
    return last_value


@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, thread_seed):
    """Accumulate in a native scalar and write the result once at the end."""
    if buffer[slot] != thread_seed:
        return
    local_counter = buffer[slot]
    for _ in range(iters):
        local_counter += np.uint64(1)
    buffer[slot] = local_counter


def aligned_uint64_buffer(length: int, alignment: int):
    """Allocate a C-contiguous uint64 view whose first byte is aligned."""
    alignment = max(64, int(alignment))
    raw = np.empty(length * np.dtype(np.uint64).itemsize + alignment - 1,
                    dtype=np.uint8)
    offset = (-raw.ctypes.data) % alignment
    view = raw[offset : offset + length * 8].view(np.uint64)
    if view.ctypes.data % 64 != 0 or not view.flags.c_contiguous:
        raise RuntimeError("Failed to allocate a contiguous 64-byte-aligned buffer.")
    return raw, view


def _resource_snapshot():
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "voluntary": int(usage.ru_nvcsw),
        "involuntary": int(usage.ru_nivcsw),
    }


def _pin_current_thread(cpu_id):
    if cpu_id is None or not hasattr(os, "sched_setaffinity"):
        return {"requested_cpu": cpu_id, "pinned": False}
    try:
        # pid 0 addresses the calling OS thread on Linux.
        os.sched_setaffinity(0, {cpu_id})
        return {"requested_cpu": cpu_id, "pinned": True}
    except (OSError, PermissionError):
        return {"requested_cpu": cpu_id, "pinned": False}


def _run_four_threads(kernel, buffer, stride, iterations, thread_seeds, context):
    thread_state = [{} for _ in range(NUM_THREADS)]
    errors = []
    start_barrier = threading.Barrier(NUM_THREADS + 1)
    assigned = context["assigned_cpu_ids"]

    def entry(thread_index):
        cpu_id = assigned[thread_index] if assigned is not None else None
        thread_state[thread_index] = _pin_current_thread(cpu_id)
        slot = thread_index * stride
        try:
            start_barrier.wait()
            kernel(buffer, slot, iterations, thread_seeds[thread_index])
        except BaseException as error:  # propagate worker errors to the main thread
            errors.append((thread_index, repr(error)))

    threads = [
        threading.Thread(target=entry, args=(thread_index,),
                         name=f"cache-worker-{thread_index}")
        for thread_index in range(NUM_THREADS)
    ]
    for thread in threads:
        thread.start()

    usage_before = _resource_snapshot()
    started_ns = time.perf_counter_ns()
    start_barrier.wait()
    for thread in threads:
        thread.join()
    elapsed_ns = time.perf_counter_ns() - started_ns
    usage_after = _resource_snapshot()

    if errors:
        raise RuntimeError(f"Worker failure(s): {errors}")

    if usage_before is None or usage_after is None:
        context_switches = None
    else:
        context_switches = {
            "voluntary": usage_after["voluntary"] - usage_before["voluntary"],
            "involuntary": usage_after["involuntary"] - usage_before["involuntary"],
        }
    return elapsed_ns, context_switches, thread_state


def _line_groups(buffer, stride, line_size):
    groups = {}
    base = int(buffer.ctypes.data)
    base_line = base // line_size
    for thread_index in range(NUM_THREADS):
        slot = thread_index * stride
        line = (base + slot * 8) // line_size - base_line
        groups.setdefault(str(line), []).append(thread_index)
    return groups


def _prepare_buffer(stride, line_size, thread_seeds):
    length = (NUM_THREADS - 1) * stride + 1
    raw, buffer = aligned_uint64_buffer(length, line_size)
    for thread_index, seed in enumerate(thread_seeds):
        buffer[thread_index * stride] = seed
    return raw, buffer


def _measure_layout(stride, context, iterations, thread_seeds):
    raw, buffer = _prepare_buffer(stride, context["cache_line_size_bytes"],
                                  thread_seeds)
    elapsed_ns, switches, thread_state = _run_four_threads(
        worker_strided_accumulate, buffer, stride, iterations, thread_seeds, context
    )
    values = [int(buffer[index * stride]) for index in range(NUM_THREADS)]
    expected = [(int(seed) + iterations) & UINT64_MASK for seed in thread_seeds]
    parity_ok = values == expected
    total_updates = NUM_THREADS * iterations
    return {
        "stride_elements": stride,
        "spacing_bytes": stride * 8,
        "elapsed_ns": int(elapsed_ns),
        "ns_per_update": elapsed_ns / total_updates,
        "total_updates": total_updates,
        "context_switches": switches,
        "thread_affinity": thread_state,
        "cache_line_groups": _line_groups(
            buffer, stride, context["cache_line_size_bytes"]
        ),
        "final_counters": values,
        "expected_counters": expected,
        "parity_passed": parity_ok,
        "base_address_64_byte_aligned": int(buffer.ctypes.data) % 64 == 0,
        "base_address_cache_line_aligned": (
            int(buffer.ctypes.data) % context["cache_line_size_bytes"] == 0
        ),
    }


def _measure_private(context, iterations, thread_seeds):
    line_size = context["cache_line_size_bytes"]
    baseline_stride = max(8, (line_size + 7) // 8)
    raw, buffer = _prepare_buffer(baseline_stride, line_size, thread_seeds)
    elapsed_ns, switches, thread_state = _run_four_threads(
        worker_private_accumulate,
        buffer,
        baseline_stride,
        iterations,
        thread_seeds,
        context,
    )
    values = [int(buffer[index * baseline_stride]) for index in range(NUM_THREADS)]
    expected = [(int(seed) + iterations) & UINT64_MASK for seed in thread_seeds]
    parity_ok = values == expected
    total_updates = NUM_THREADS * iterations
    return {
        "stride_elements": baseline_stride,
        "spacing_bytes": baseline_stride * 8,
        "elapsed_ns": int(elapsed_ns),
        "ns_per_update": elapsed_ns / total_updates,
        "total_updates": total_updates,
        "context_switches": switches,
        "thread_affinity": thread_state,
        "cache_line_groups": _line_groups(
            buffer, baseline_stride, line_size
        ),
        "final_counters": values,
        "expected_counters": expected,
        "parity_passed": parity_ok,
        "base_address_64_byte_aligned": int(buffer.ctypes.data) % 64 == 0,
        "base_address_cache_line_aligned": int(buffer.ctypes.data) % line_size == 0,
    }


def _aggregate_trials(samples):
    last = samples[-1]
    durations = [int(sample["elapsed_ns"]) for sample in samples]
    switches = [sample["context_switches"] for sample in samples]
    valid_switches = [value for value in switches if value is not None]
    if valid_switches:
        median_switches = {
            key: statistics.median(value[key] for value in valid_switches)
            for key in ("voluntary", "involuntary")
        }
    else:
        median_switches = None
    median_ns = int(statistics.median(durations))
    return {
        **last,
        "elapsed_ns": median_ns,
        "ns_per_update": median_ns / last["total_updates"],
        "context_switches": median_switches,
        "trial_elapsed_ns": durations,
        "trial_context_switches": switches,
        "trial_min_ns": min(durations),
        "trial_max_ns": max(durations),
        "trial_parity_passed": [bool(sample["parity_passed"]) for sample in samples],
        "parity_passed": all(sample["parity_passed"] for sample in samples),
        "trial_cache_line_groups": [sample["cache_line_groups"] for sample in samples],
    }


def run_stride_benchmark(context, iterations=ITERATIONS_PER_THREAD):
    """Run four contended layouts and the register-private baseline."""
    seed0 = context["instance_seed"]
    thread_seeds = np.array(
        [((seed0 + i * 0x9E3779B97F4A7C15) & UINT64_MASK)
         for i in range(NUM_THREADS)],
        dtype=np.uint64,
    )

    # Compile each Numba signature before timing; these one-iteration calls are
    # excluded from benchmark measurements.
    warm = np.zeros(1, dtype=np.uint64)
    worker_strided_accumulate(warm, 0, 1, np.uint64(0))
    worker_private_accumulate(warm, 0, 1, np.uint64(0))

    trial_rows = {stride: [] for stride in STRIDES}
    private_rows = []
    for trial_index in range(BENCHMARK_TRIALS):
        offset = trial_index % len(STRIDES)
        order = STRIDES[offset:] + STRIDES[:offset]
        for stride in order:
            row = _measure_layout(stride, context, iterations, thread_seeds)
            trial_rows[stride].append(row)
            print(
                f"trial={trial_index + 1} stride={stride:2d} "
                f"elapsed={row['elapsed_ns']} ns "
                f"vol_cs={None if row['context_switches'] is None else row['context_switches']['voluntary']} "
                f"invol_cs={None if row['context_switches'] is None else row['context_switches']['involuntary']}",
                flush=True,
            )
        private_row = _measure_private(context, iterations, thread_seeds)
        private_rows.append(private_row)
        print(
            f"trial={trial_index + 1} register_private="
            f"{private_row['elapsed_ns']} ns",
            flush=True,
        )

    layouts = {
        f"stride_{stride}": _aggregate_trials(trial_rows[stride])
        for stride in STRIDES
    }
    private = _aggregate_trials(private_rows)
    for stride in STRIDES:
        row = layouts[f"stride_{stride}"]
        print(
            f"stride={stride:2d} median={row['elapsed_ns']} ns "
            f"range=[{row['trial_min_ns']}, {row['trial_max_ns']}] ns "
            f"ns/update={row['ns_per_update']:.6f}",
            flush=True,
        )
    print(
        f"register_private median={private['elapsed_ns']} ns "
        f"range=[{private['trial_min_ns']}, {private['trial_max_ns']}] ns "
        f"ns/update={private['ns_per_update']:.6f}",
        flush=True,
    )

    stride1 = layouts["stride_1"]["elapsed_ns"]
    stride8 = layouts["stride_8"]["elapsed_ns"]
    summary = {
        "layouts": layouts,
        "register_private": private,
        "thread_seeds": [int(value) for value in thread_seeds],
        "iterations_per_thread": int(iterations),
        "trials": BENCHMARK_TRIALS,
        "slowdown_stride1_vs_stride8": stride1 / stride8 if stride8 else None,
        "register_private_vs_stride1_speedup": (
            stride1 / private["elapsed_ns"] if private["elapsed_ns"] else None
        ),
    }
    print(
        "slowdown_stride1_vs_stride8="
        f"{summary['slowdown_stride1_vs_stride8']:.6f}",
        flush=True,
    )
    return summary


def _get_llvm_text(dispatcher):
    llvm = dispatcher.inspect_llvm()
    if isinstance(llvm, dict):
        return "\n".join(str(value) for value in llvm.values())
    return str(llvm)


def audit_llvm_and_parity(benchmark, context):
    """Audit native volatile i64 memory operations and exact result parity."""
    llvm_text = _get_llvm_text(worker_strided_accumulate)
    native_candidates = re.findall(
        r"(?ms)^define[^{]*worker_strided_accumulate[^{]*\{.*?^\}", llvm_text
    )
    native_functions = [
        function for function in native_candidates if "@cpython" not in function
    ]
    if not native_functions:
        raise RuntimeError("Could not locate the native worker function in LLVM IR.")
    native_ir = native_functions[0]
    private_llvm = _get_llvm_text(worker_private_accumulate)
    private_candidates = re.findall(
        r"(?ms)^define[^{}]*worker_private_accumulate[^{}]*\{.*?^\}",
        private_llvm,
    )
    private_functions = [
        function for function in private_candidates if "@cpython" not in function
    ]
    if not private_functions:
        raise RuntimeError("Could not locate the private baseline in LLVM IR.")
    private_native_ir = private_functions[0]
    lines = [line.strip() for line in native_ir.splitlines()]
    private_lines = [line.strip() for line in private_native_ir.splitlines()]
    volatile_loads = [line for line in lines if re.search(r"load volatile i64\b", line)]
    volatile_stores = [line for line in lines if re.search(r"store volatile i64\b", line)]
    native_i64_stores = [
        line for line in lines if re.search(r"\bstore(?: volatile)? i64\b", line)
    ]
    python_object_fallback = any(
        marker in native_ir for marker in ("PyObject", "Py_INCREF", "Py_DECREF")
    )
    private_stores = [
        line for line in private_lines if re.search(r"\bstore i64\b", line)
    ]
    private_adds = [line for line in private_lines if re.search(r"\badd i64\b", line)]
    private_object_fallback = any(
        marker in private_native_ir for marker in ("PyObject", "Py_INCREF", "Py_DECREF")
    )

    all_results = [*benchmark["layouts"].values(), benchmark["register_private"]]
    baseline = benchmark["register_private"]["final_counters"]
    parity = all(
        result["final_counters"] == baseline and result["parity_passed"]
        for result in all_results
    )
    llvm_ok = bool(volatile_loads and volatile_stores and native_i64_stores)
    aligned = all(result["base_address_64_byte_aligned"] for result in all_results)
    line_aligned = all(
        result["base_address_cache_line_aligned"] for result in all_results
    )
    private_ok = len(private_stores) == 1 and bool(private_adds)

    return {
        "parity_passed": bool(parity),
        "all_final_counters_bit_exact": bool(parity),
        "llvm_native_i64_load_count": len(volatile_loads),
        "llvm_native_i64_store_count": len(volatile_stores),
        "llvm_any_i64_store_count": len(native_i64_stores),
        "llvm_has_volatile_load_and_store": bool(llvm_ok),
        "python_object_fallback_detected": bool(python_object_fallback),
        "llvm_audit_passed": bool(llvm_ok and not python_object_fallback),
        "private_baseline_i64_store_count": len(private_stores),
        "private_baseline_i64_add_count": len(private_adds),
        "private_baseline_one_store_verified": private_ok,
        "private_baseline_ir_examples": private_adds[:2] + private_stores[:2],
        "private_python_object_fallback_detected": bool(private_object_fallback),
        "all_buffers_64_byte_aligned": bool(aligned),
        "all_buffers_cache_line_aligned": bool(line_aligned),
        "volatile_ir_examples": (volatile_loads[:2] + volatile_stores[:2]),
        "verification_passed": bool(
            parity
            and llvm_ok
            and not python_object_fallback
            and not private_object_fallback
            and private_ok
            and aligned
            and line_aligned
        ),
    }


def run_main(git_user: str, iterations: int, manifest_path: str):
    context = init_system_context(git_user)
    print("SYSTEM_CONTEXT=" + json.dumps(context, sort_keys=True), flush=True)
    benchmark = run_stride_benchmark(context, iterations)
    audit = audit_llvm_and_parity(benchmark, context)

    manifest = {
        "schema_version": 1,
        "runtime_context": context,
        "thread_count": NUM_THREADS,
        "iterations_per_thread": int(iterations),
        "required_iterations_per_thread": ITERATIONS_PER_THREAD,
        "full_workload": int(iterations) == ITERATIONS_PER_THREAD,
        "benchmarks": benchmark,
        "verification": audit,
        "verification_passed": bool(
            audit["verification_passed"] and int(iterations) == ITERATIONS_PER_THREAD
        ),
    }

    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print("LLVM_AUDIT=" + json.dumps(audit, sort_keys=True), flush=True)
    print(f"verification_passed={manifest['verification_passed']}", flush=True)
    print(f"Wrote {manifest_path}", flush=True)
    if not manifest["verification_passed"]:
        raise RuntimeError(
            "Benchmark verification failed; see cache_lab_manifest.json."
        )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default=GIT_USERNAME)
    parser.add_argument("--iterations", type=int, default=ITERATIONS_PER_THREAD)
    parser.add_argument("--manifest", default="cache_lab_manifest.json")
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_main(args.username, args.iterations, args.manifest)


if __name__ == "__main__":
    main()
