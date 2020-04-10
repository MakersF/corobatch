# Fused Multiply Add Benchmark

The benchmark compares the performances of using `corobatch` to execute a fused multiply add on a vector of floats.

The comparison is with a raw loop.

This is an unfavorable use case for the library, because:

- the operation to execute is very fast (in the order of nanoseconds)
- the compiler can detect the pattern and automatically perform vectorization

Here we are going to see the impact of the overhead the library introduces.

On my machine, these are the results

```
$ corobatch/benchmark/multiply_add/benchmark_multiply_add
2020-04-08 22:15:52
Running corobatch/benchmark/multiply_add/benchmark_multiply_add
Run on (4 X 2700 MHz CPU s)
CPU Caches:
L1 Data 32 KiB (x2)
L1 Instruction 32 KiB (x2)
L2 Unified 256 KiB (x2)
L3 Unified 3072 KiB (x1)
Load Average: 0.65, 0.41, 0.52
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
----------------------------------------------------------------
Benchmark Time CPU Iterations
----------------------------------------------------------------
BM_fma_loop/8            8.64 ns    8.64 ns 79151233
BM_fma_loop/64           62.7 ns    62.7 ns 10865126
BM_fma_loop/512           575 ns     575 ns  1194952
BM_fma_loop/4096         4683 ns    4679 ns   151737
BM_fma_loop/8192         9264 ns    9264 ns    75584
BM_fma_corobatch/8       1788 ns    1788 ns   397311
BM_fma_corobatch/64     11201 ns   11189 ns    61833
BM_fma_corobatch/512    82016 ns   82013 ns     8385
BM_fma_corobatch/4096  678026 ns  678008 ns     1016
BM_fma_corobatch/8192 1325141 ns 1325140 ns      527
```

The overhead the library adds on the laptop on which it has been executed is roughly 200ns per task, or ~150x longer than the simple loop.

The biggest impact on the result is given by the allocations needed for several components>
- the coroutine context
- the batch state
- bookkeeping information

The `performance` branch has some changes to address that which result in a 2x improvement.
