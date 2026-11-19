# False-Sharing Laboratory: Analysis and Run Results

## Recorded Environment

The benchmark was run with four native worker threads, 60,000,000 iterations per
thread, and three trials per layout. Reported durations below are the median of
those three trials; the individual measurements are retained in
[`cache_lab_manifest.json`](cache_lab_manifest.json).

| Item | Recorded value |
|---|---:|
| CPU | Apple M1 (`arm64`) |
| Logical CPU count | 8 |
| Detected L1 cache-line size | 128 bytes |
| CPU affinity | Not available through this macOS runtime |
| Python | CPython 3.13.0, GIL enabled |
| Numba / NumPy | 0.68.1 / 2.5.5 |
| Iterations per worker | 60,000,000 |
| Trials per layout | 3 |

The detected 128-byte cache line is important: it differs from the 64-byte line
assumed by the generic layout table in the laboratory specification.

## Question 1 — MESI State Transitions

Assume both cores have read the same cache line, so each begins in the Shared
state. `buffer[0]` and `buffer[1]` are separate 8-byte values, but they occupy
the same cache line under stride 1.

1. Core 0 stores to `buffer[0]`. It cannot modify a Shared line, so it issues a
   Read-For-Ownership (RFO) request. Once ownership is granted, Core 0 changes
   from Shared to Modified and Core 1's copy changes from Shared to Invalid.
2. Core 1 immediately stores to `buffer[1]`. Its copy is Invalid, so it also
   requests ownership. The request invalidates Core 0's Modified copy; Core 1
   receives the line in Modified state.
3. When Core 0 next stores to `buffer[0]`, its copy is Invalid and it must issue
   another RFO. The line changes ownership again. Repeated stores therefore
   move the same line between cores even though the two cores write different
   addresses.

The store can wait in a core's store buffer while ownership is pending. The
store cannot be completed in that core's L1 until the RFO is granted. The
resulting ownership transfers and invalidations are the false-sharing cost.
On this measured Apple M1, the detected coherence line is 128 bytes, so the
same two values also share a line there.

## Question 2 — Measured Slowdown and Update Cost

Each timing covers four workers and 240,000,000 logical updates in total. The
per-update column is the median wall time divided by that logical update count.

| Layout | Stride | Median wall time (ns) | Three-trial range (ns) | ns per logical update | Median involuntary context switches |
|---|---:|---:|---:|---:|---:|
| Contiguous | 1 | 107,844,500 | 88,438,875–117,101,083 | 0.449352083 | 427 |
| Half-line | 4 | 192,650,875 | 136,841,500–230,643,875 | 0.802711979 | 2,108 |
| 64-byte spacing | 8 | 39,625,583 | 24,056,708–60,817,042 | 0.165106596 | 302 |
| 128-byte spacing | 16 | 30,676,167 | 30,141,958–33,621,250 | 0.127817363 | 430 |
| Register-private baseline | 16 | 183,875 | 103,208–441,709 | 0.000766146 | 16 |

Voluntary context switches were zero in every recorded trial. The measured
stride-1 slowdown relative to stride 8 is **2.721588×**. The register-private
baseline was **586.509857×** faster than the stride-1 median.

The per-update baseline value is a normalized throughput figure, not the time
of one executed increment. As shown in Question 4, LLVM reduced the private
counter loop to one scalar addition and one final store.

## Question 3 — Adjacent-Line Prefetching and 128-Byte Pairing

The stride-8 median was 39,625,583 ns; the stride-16 median was 30,676,167 ns.
Stride 16 was **22.584945% faster** in this run. The detected 128-byte line
explains the address grouping recorded by the program:

- Stride 8 places workers 0 and 1 in one 128-byte line, and workers 2 and 3 in
  another. It therefore still has two-way false sharing on this host.
- Stride 16 places each worker in a separate 128-byte line.

An adjacent-line prefetcher may fetch two neighboring 64-byte lines together as
a 128-byte sector. That affects residency and traffic, but a read prefetch by
itself does not grant write ownership or invalidate another core's line. A
64-byte spacing can still suffer ownership invalidations when the machine's
coherence granularity is 128 bytes, as detected here, or when object alignment
causes two writers to land in the same coherence unit. The individual
stride-8 trials varied from 24.1 ms to 60.8 ms; macOS did not expose the
per-thread affinity interface used by the benchmark, so scheduling noise is a
material limitation of this comparison.

## Question 4 — Register Promotion and LLVM IR

The native specialization of `worker_strided_accumulate.inspect_llvm()` contains
one static volatile 64-bit load and one static volatile 64-bit store in the
loop body:

```llvm
%.242 = load volatile i64, ptr %0, align 8
store volatile i64 %.243, ptr %0, align 8
```

The manifest records one `load volatile i64` and one `store volatile i64` in the
native worker loop body, with no Python-object fallback. The manifest also
counts two `i64` stores in the full native function: the volatile counter store
and the store of the function's returned value. Each static loop instruction
is executed on every iteration, so each worker performs 60,000,000 native
read-modify-write updates. The volatile accesses prevent LLVM from retaining
the counter only in an SSA register and writing it once after the loop.

The private baseline has one native `add i64` and one `store i64` in its
optimized function body. LLVM folded the local scalar loop into a single
addition, which explains its very small measured time. This is why its
normalized nanoseconds per logical update should not be interpreted as the
latency of a physical increment.

## Question 5 — CPython GIL and `nogil=True`

In this run, CPython reported that the GIL was enabled. In a standard GIL build,
only one thread executes Python bytecode at a time. A pure-Python CPU-bound
thread test would therefore serialize its increments and add interpreter lock
handoffs; it would not create the concurrent native stores needed to expose
cache-line ping-pong. The benchmark's workers are Numba-compiled with
`nogil=True`, so they execute native code concurrently while sharing the same
NumPy buffer. This makes the cache-coherence behavior observable.

The pure-Python GIL-only comparison was not separately timed in this run; the
explanation above is based on CPython's documented GIL behavior and the
recorded `gil_enabled: true` runtime field.

## Verification

The run recorded `verification_passed: true`: all five final counter vectors
were bit-exact, the 64-bit buffers were aligned, the strided worker's LLVM IR
contained volatile 64-bit loads and stores, and the private baseline contained
one final 64-bit counter store. Full telemetry is in
[`cache_lab_manifest.json`](cache_lab_manifest.json).

## References

- [CPython 3.13 threading and GIL documentation](https://docs.python.org/3.13/library/threading.html)
- [Numba `nogil=True` documentation](https://numba.readthedocs.io/en/stable/user/jit.html)
- [Numba intrinsic extension API](https://numba.readthedocs.io/en/stable/extending/high-level.html)
