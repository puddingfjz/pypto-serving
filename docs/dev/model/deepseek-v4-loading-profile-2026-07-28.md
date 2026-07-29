# DeepSeek V4 Startup Loading Profile

This document records one cold-start profile of DeepSeek V4 Flash W8A8 on
eight Ascend devices. It separates measured trace spans from inferred
intervals, explains the dependency graph, and lists startup optimization
opportunities.

The numbers come from one run and are a baseline for investigation, not a
stable performance guarantee. Every optimization needs repeated cold and warm
runs before a speedup is claimed.

## Measurement and Results

| Item | Value |
| --- | --- |
| Date | 2026-07-28 |
| pypto-serving | `627df90` |
| PyPTO | `340d6c567` |
| Simpler runtime | `9922afdb` |
| Checkpoint | `/data/models/dsv4-flash-w8a8` |
| Devices | 8 Ascend devices, IDs 8-15 |
| Layout | attention DP=8, MoE EP=8 |
| MTP | enabled |
| Trace | `/tmp/pypto-serving-deepseek-load-627df90-run4/trace.json` |
| Workload | `test_deepseek_v4_http_completion_matches_expected_text` |

The server completed model registration and became healthy. Its first prefill
then failed because `Worker.copy_to()` rejected an interior pointer into a live
decode-cache allocation. Registration had already completed, so the startup
intervals remain valid, but a future benchmark must also complete one prefill
and decode request.

Model registration took **545.220 seconds**:

```text
PyptoExecutor.register_model                         545.220 s
├─ compile, compile preparation, and runner setup    300.151 s
└─ PyptoExecutor.preflight                           245.069 s
```

`preflight` is a parent span. Weight loading, worker creation, and resident
weight upload are already included in it.

| Category | Time (s) | Registration | Source |
| --- | ---: | ---: | --- |
| MTP dummy-argument preparation | 231.803 | 42.5% | inferred gaps |
| Four `jit_fn.compile()` calls | 68.213 | 12.5% | trace |
| Main-model weight load and pack | 108.129 | 19.8% | trace |
| Persistent L3 worker construction | 92.271 | 16.9% | trace |
| Main-model resident weight upload | 32.571 | 6.0% | trace |
| MTP resident weight upload | 1.089 | 0.2% | trace |
| Remaining registration and preflight work | 11.144 | 2.0% | difference |
| **Total** | **545.220** | **100.0%** | |

The four JIT spans were:

| Program | Time (s) |
| --- | ---: |
| Main prefill | 27.539 |
| Main decode | 27.585 |
| MTP prefill | 6.567 |
| MTP decode | 6.522 |
| **Total** | **68.213** |

The gaps immediately before the MTP compile calls were:

| Inferred interval | Time (s) |
| --- | ---: |
| MTP prefill dummy arguments | 151.170 |
| MTP decode dummy arguments | 80.633 |
| **Total** | **231.803** |

`_mtp_dummy_args()` performs module/spec work and creates real CPU
`torch.empty()` tensors before `_compile_l3_callable()` enters its profile
span. The trace cannot yet separate import, spec construction, and allocation.

Preflight was:

```text
PyptoExecutor.preflight                              245.069 s
├─ host/global/buffer preparation                      8.531 s
├─ main-model weight load and pack                    108.129 s
├─ small preparation gap                                0.060 s
├─ upload_resident_weights                            125.940 s
│  ├─ create persistent L3 worker                      92.271 s
│  ├─ upload resident main weights                     32.571 s
│  ├─ upload resident MTP weights                       1.089 s
│  └─ wrapper overhead                                  0.009 s
└─ resident-cache materialization and final work        2.409 s
```

The notable smaller preparation spans were MTP buffers at 2.947 seconds,
decode work cache at 3.835 seconds, and LM head at 1.576 seconds.

## Optimization Experiments

Later experiments used pypto-serving `e57e14f`, PyPTO `478ddad5a`, the
repository-pinned Simpler runtime, pypto-lib `7e7d4cc`, PTOAS 0.48, and device
IDs `0,2,4,6,8,10,12,14`. The corresponding serial control registered the
model in 506.758 seconds. Its main layer load and pack took 90.642 seconds,
persistent worker construction took 86.593 seconds, main weight upload took
25.032 seconds, and MTP upload took 0.769 seconds.

Successful experiments were kept on separate branches. Their savings are not
additive because they remove or hide overlapping portions of startup.

| Optimization | Registration result | Change from control | Status |
| --- | ---: | ---: | --- |
| Compile from tensor signatures | 283.051 s | -223.708 s (-44.1%) | kept |
| Reuse compiled artifacts | 310.674 s | -196.084 s (-38.7%) | kept |
| Load prepacked layer sidecar | 447.542 s | -59.217 s (-11.7%) | kept |

Compiling from signatures removed materialization of the large dummy tensors
used only to describe the four program inputs. The individual JIT spans still
took 68.142 seconds, but the inferred MTP argument preparation disappeared.
The complete registration trace was
`/tmp/pypto-serving-deepseek-signature-latest`.

The compiled-artifact cache saved 150.633 seconds directly in JIT work and
196.084 seconds in complete registration on a warm launch. It copies the
fully assembled callable output directories as well as compiled programs, so
worker construction can reuse device binaries across process launches.

The prepacked sidecar stores the final rank-stacked layout consumed by resident
upload. A hot page-cache run reduced registration to 447.542 seconds. Creating
the 322.818 GiB sidecar is an offline operation that took 577.431 seconds. A
Linux `mincore()` sample prevents a cold sidecar from turning upload into
random page faults: when fewer than 95 percent of sampled pages are resident,
startup uses the original checkpoint loader. The gate took about 0.02 seconds.

### Combined Experiment

Commit `d0cef65` combines signature-only compilation, the persistent callable
cache, and the prepacked weight sidecar. The merge also fingerprints cache
entries from the evaluated JIT function signature and scalar arguments, so a
cache lookup does not reconstruct the deleted dummy tensors.

Four launches covered cache publication, a warm cache with a cold sidecar, and
two warm-cache/hot-sidecar repetitions:

| Cache and sidecar state | Devices | Registration (s) | Layer load + pack (s) | Worker (s) | Main upload (s) | MTP upload (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Miss + publish, cold sidecar fallback | `0,2,4,6,8,10,12,14` | 349.297 | 95.924 | 139.722 | 25.914 | 1.037 |
| Hit, cold sidecar fallback | `0,2,4,6,8,10,12,14` | 276.206 | 223.768 | 3.459 | 25.010 | 0.882 |
| Hit, hot sidecar, run 1 | `0,1,2,3,4,5,6,7` | 94.058 | 0.000 | 3.990 | 53.879 | 1.227 |
| Hit, hot sidecar, run 2 | `0,1,2,3,4,5,6,7` | 87.133 | 0.000 | 3.128 | 58.795 | 0.813 |

The two hot repetitions averaged **90.595 seconds**, which is 416.163 seconds
or 82.1 percent below the 506.758-second control (5.59x as fast). Their trace
directories are
`/tmp/pypto-serving-deepseek-combined-hit-hot-1-d0-7-20260729` and
`/tmp/pypto-serving-deepseek-combined-hit-hot-2-d0-7-20260729`.

The cache-miss launch includes copying four complete callable output trees into
the cache. That publication work and shared-machine load increased worker
construction to 139.722 seconds, so it is not a steady-state result. The
cache-hit/cold-sidecar launch proves the cache path but is also not a clean
absolute comparison: concurrent Host work stretched layer packing from the
control's 90.642 seconds to 223.768 seconds. Its compile spans disappeared and
worker construction fell to 3.459 seconds.

All four launches completed resident-weight upload, then exited while sizing
the KV cache because residual device allocations left about 6.2 GB free on
device 0. The 90 percent HBM budget could not fit one additional 312.7 MB cache
slot. `PyptoExecutor.register_model` closes its trace span on this exception,
so the startup intervals above are complete, but these launches do not replace
an end-to-end generation correctness test.

Warming the sidecar was itself expensive under concurrent machine load. A
sequential read of all 322.818 GiB took 577.710 seconds and left 92.383 percent
sampled residency because other workloads reclaimed pages during the read. A
second pass took 155.860 seconds, after which `fincore` reported all
84,624,906 pages resident. These costs are not included in hot-start time; the
sidecar optimization requires an already resident file and deliberately falls
back otherwise.

The hot sidecar also explains why main upload is slower than the control.
`safe_open().get_tensor()` returns tensors backed by a lazy file mapping.
`mincore()` proves that file contents are in the Linux page cache, but it does
not install this process's page-table entries or prefault/pin every page for
the device copy. The first `alloc_stacked_tensor()` upload therefore touches
roughly 84.6 million 4 KiB pages and pays minor-fault and mapping/pinning
overhead. The original pack path has already written every anonymous
destination page, so its mappings are hot before upload. This moves main
upload from about 25 seconds to 50--59 seconds, but avoiding roughly 91 seconds
of packing still gives a net startup improvement. A separate prefault pass
would only move that cost unless it can safely overlap other startup work.

The following experiments were rejected and left no code branch:

- Four-way blind shard prefetch read 270 of 347 GB in 294.689 seconds before
  cancellation. Registration increased to 514.419 seconds.
- A depth-one checkpoint prefetch pipeline was unstable and did not produce a
  repeatable improvement.
- Keeping safetensors readers open cannot reduce repeated shard opens for this
  checkpoint. The 43 hidden layers map one-to-one to 43 distinct shard files,
  and `load_many()` already opens each layer's single shard only once.
- Rank-parallel resident upload plus releasing the GIL around device control
  increased main and MTP upload from 24.800 + 0.812 seconds to
  25.967 + 1.020 seconds.
- Compact rank-local Host weights saved 56.42 GiB of physical Host memory but
  increased preparation plus upload from 115.674 to 115.862 seconds.
- Packing multiple complete layers concurrently multiplied the roughly 8 GiB
  per-layer intermediate working set and regressed through memory-bandwidth
  contention.
- Overlapping serial layer packing with independent RoPE, LM-head, and shared
  Host-buffer preparation was functionally safe but also contended for memory
  bandwidth. A Host-only matched comparison increased total preparation from
  146.098 to 178.588 seconds. Layer packing itself increased from 141.011 to
  173.236 seconds, overwhelming the roughly 4.19 seconds of overlapped work.
- Preparing worker executable artifacts concurrently with weight packing
  stretched the pack interval to roughly 142 seconds. All eight children
  reached their ready state, but the first resident materialization then
  stalled. The experiment was removed rather than carrying a new artifact
  lifecycle API without a safe end-to-end result.

A later full serial control under heavier machine load measured 141.771
seconds for layer packing, consistent with the 141.011-second Host-only serial
sample. Its worker initialization could not complete because unrelated
processes left insufficient device memory for the 8 GiB static arena, so that
run is not used as a complete registration result.

### What "Persistent Worker Creation" Includes

The 92.271-second span is not just process creation.
`DistributedWorker.__init__()` first:

1. visits all four distributed programs;
2. calls `_assemble_chip_callables()` for each chip sub-build;
3. runs `compile_and_assemble()` to compile kernels and orchestration C++;
4. creates `ChipCallable` objects and loads host orchestration;
5. registers every callable.

It then forks eight chip workers. Each child initializes CANN and the device
runtime, prewarms the runtime arena, prepares every startup callable, and
publishes `INIT_READY`. The parent waits for all devices.

Wall-clock evidence gives this approximate split:

```text
Persistent L3 worker construction                    92.271 s
├─ executable preparation, registration, and work
│  before the first child runtime starts             ~82.7 s
└─ device initialization, callable preparation,
   and the all-device readiness barrier               ~9.6 s
```

This split is inferred:

- the worker span began at approximately `01:07:47.0`;
- kernel cache files were written through `01:09:05.8`;
- the first child CANN runtime started at `01:09:09.7`;
- the worker span ended at approximately `01:09:19.3`.

Each program contains its own `lm_head` sub-build, so similar content is
assembled repeatedly in separate timestamped output directories. The outer
program/sub-build loop is sequential, while each `compile_and_assemble()` call
creates an internal pool of up to 64 threads.

The 68.213-second JIT total and the worker's executable preparation are
different compilation layers. The former covers graph/JIT/code generation;
executable kernel and orchestration compilation continues during worker
construction.

The worker span does not upload the 346.6 GB main-model weights. That happens
in the following 32.571-second span.

## Dependency and Parallelization Design

The current startup is mostly one chain:

```text
metadata validation
  → build four program inputs and JIT programs
  → allocate host buffers
  → load and pack all weights
  → compile and assemble executable callables
  → fork and initialize the worker
  → upload resident weights
  → ready
```

A target dependency graph can expose three branches:

```text
metadata, layout, and weight-map validation
├─ program: frontend → backend compile → executable assembly
├─ weights: global/MTP preparation + main-layer IO and packing
└─ buffers: independent host/shared allocation

join: all artifacts and fork-visible tensors are ready
  → stop and join every background executor
  → fork and initialize chip workers
  → upload resident weights
  → ready
```

The pre-fork join is mandatory. All inherited weights, shared buffers, and
startup callables must exist before `Worker.init()`. No loading or compilation
thread may remain active across `os.fork()`, because a child could inherit
locked library state.

An unsafe large operation should be decomposed rather than either kept fully
serial or made fully concurrent.

### MTP Argument Decomposition

The conflict is `_deepseek_v4_import_context()`, which mutates `sys.argv`,
`sys.path`, and selected `sys.modules` entries.

1. Serialize only module import and `build_tensor_specs()`.
2. Copy results into immutable `(name, shape, dtype)` descriptors.
3. Restore global import state.
4. Materialize independent arguments concurrently outside the context.
5. Join only the arguments required by each compile.

Prefer eliminating materialization: compile from symbolic, fake, meta, or
explicit tensor descriptors if PyPTO supports that contract. A longer-term
pypto-lib change should replace command-line-driven module globals with an
explicit immutable configuration.

### Compiler and Assembly Decomposition

1. Serialize frontend capture only where it uses global Python state.
2. Publish immutable per-program IR in unique output directories.
3. Put all ready backend, kernel, PTOAS, and orchestration jobs into one shared
   bounded pool.
4. Assemble each callable when its own dependencies complete.
5. Serialize only generated-module publication if it still mutates import
   state.

Do not run four outer tasks that each create another 64-thread pool. Flatten
the nested pools into one scheduler with CPU, memory, and toolchain limits.

Split `DistributedWorker` into an artifact-preparation step and a short
worker-start step. Artifact preparation can overlap weights and later program
compilation; worker start must remain behind the pre-fork join.

### Weight-Loading Decomposition

Full-layer thread-pool packing was already tried and regressed. Each layer can
create about 8 GB of intermediates, so multiple layers multiply peak memory and
contend for memory bandwidth.

A bounded pipeline remains possible:

1. Allocate final stacked destinations and layer offsets serially.
2. Prefetch checkpoint data for layer `N+1` while packing layer `N`.
3. Limit prefetch depth by measured memory headroom.
4. Transform disjoint tensor groups concurrently only when they write disjoint
   final views.
5. Release raw layer storage before advancing.
6. Join the full host weight set before fork.

Reads can also be grouped by safetensors file to reduce repeated open and
metadata work.

### Device and Upload Decomposition

The eight chip workers are already processes, but detailed milestones are
missing. Separate parent fork delay, per-device CANN init, arena prewarm,
callable preparation, and the final readiness barrier before optimizing them.

`alloc_stacked_tensor()` currently uploads rank shards in a blocking loop.
A future batch/asynchronous API can:

1. allocate independent device destinations;
2. issue copies concurrently across distinct devices;
3. bound total in-flight bytes;
4. join all ranks before publishing a `StackedDeviceTensor`;
5. preserve rollback state until every copy succeeds;
6. release inherited host references only after all uploads complete.

Calling current worker methods from arbitrary threads should not be assumed
safe.

## Optimization Inventory

Potential savings are not additive. An overlap can hide only the shorter branch,
and eliminating work changes the later critical path.

### Measurement

- Add spans for import, spec construction, and dummy tensor allocation.
- Profile every program and chip sub-build separately.
- Split cache lookup, kernel compile, orchestration compile, and assembly.
- Record fork, CANN-ready, arena-ready, callable-ready, and `INIT_READY` per
  device.
- Record checkpoint IO, per-layer pack, allocation, and per-rank upload.
- Report cold and warm cache runs, peak RSS, IO throughput, memory bandwidth,
  device-start skew, and upload throughput.

### Remove or Reuse Work

- Replace real MTP dummy tensors with shape-only inputs.
- Cache immutable MTP tensor specs by deployment specialization.
- Replace global import configuration with explicit immutable configuration.
- Prebuild deployable distributed programs and executable artifacts.
- Use a shared content-addressed binary cache across timestamped outputs.
- Deduplicate the four LM-head sub-builds.
- Reuse common kernel binaries across prefill, decode, MTP, and LM head.
- Deduplicate startup callable preparation by callable identity.
- Precompute callable hashes and registration descriptors before startup.
- Compile only variants required by the active deployment.
- Create and validate an offline prepacked checkpoint layout.
- Persist validated packed host artifacts across restarts.
- Remove remaining dtype, contiguous, replication, stack, and reshape copies.
- Release dummy tensors, IR, compiler objects, and unused mappings before fork.

### Overlap and Pipeline Work

- Start host weight preparation after metadata validation while programs
  compile. Perfect overlap can hide at most the measured 108.129 seconds.
- Start executable assembly as soon as each program emits immutable output.
- Overlap executable preparation with weight loading; only actual worker start
  needs the completed fork-visible state.
- Move independent global, RoPE, shared-buffer, and scheduler allocation into a
  separate pre-fork branch.
- Pipeline depth-limited checkpoint prefetch with serial final-layout packing.
- Keep checkpoint files open and batch reads by file.
- Parallelize only disjoint transforms within one layer.
- Use NUMA-aware IO and destination placement after measuring topology.
- Add bounded rank-parallel resident upload.
- Fuse small device allocations and copies into larger arenas.
- Pipeline device allocation with asynchronous copy.
- Bound upload concurrency by bytes rather than tensor count.
- Overlap main and MTP upload only after main upload is optimized; MTP alone is
  just 1.089 seconds.

### Scheduling and Lifecycle

- Replace nested compile pools with one bounded dependency scheduler.
- Use atomic cache publication and process-level cache locking.
- Separate prepared artifacts from `DistributedWorker` lifecycle state.
- Build immutable registration snapshots and mailboxes before the fork loop.
- Minimize parent work between consecutive chip forks.
- Ensure all background executors are stopped before fork.
- Audit runtime-arena prewarm: deferral shortens readiness but moves latency to
  the first request.
- Audit zero initialization only for buffers proven overwritten before read.
- Consider reusing a compatible long-lived worker across model reloads only as
  a high-complexity design; dynamic registration, weight replacement, reset,
  and failure isolation must preserve the inheritance contract.

## Recommended Order

1. Add the missing inner spans and repeat cold and warm baselines.
2. Remove or replace materialized MTP dummy inputs.
3. Split executable preparation from worker startup.
4. Add shared binary caching and LM-head deduplication.
5. Overlap bounded weight preparation with the program branch.
6. Add a depth-limited checkpoint IO/pack pipeline.
7. Profile and then parallelize rank-level resident upload.
8. Optimize the remaining device-start and buffer-allocation tail.

Every benchmark must report the complete timing tree and peak memory, and must
finish model health, prefill, and decode correctness checks.
