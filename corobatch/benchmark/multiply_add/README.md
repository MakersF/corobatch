# Fused Multiply Add Benchmark

The benchmark compares the performances of using `corobatch` to execute a fused multiply add on a vector of floats.

The comparison is with a raw loop.

This is an unfavorable use case for the library, because:

- the operation to execute is very fast (in the order of nanoseconds)
- the compiler can detect the pattern and automatically perform vectorization

Here we are going to see the impact of the overhead the library introduces.

These are the results as run on a laptop with an i5-5200U

```
$ corobatch/benchmark/multiply_add/benchmark_multiply_add
2020-04-16 17:22:46
Running corobatch/benchmark/multiply_add/benchmark_multiply_add
Run on (4 X 2700 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x2)
  L1 Instruction 32 KiB (x2)
  L2 Unified 256 KiB (x2)
  L3 Unified 3072 KiB (x1)
Load Average: 0.89, 1.13, 1.16
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
---------------------------------------------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                                                                     Time             CPU   Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------
BM_fma_loop/8                                                                                                              8.82 ns         8.82 ns     76834946
BM_fma_loop/64                                                                                                             62.6 ns         62.6 ns     10431303
BM_fma_loop/512                                                                                                             581 ns          581 ns      1170473
BM_fma_loop/4096                                                                                                           4647 ns         4646 ns       149500
BM_fma_loop/8192                                                                                                           9369 ns         9369 ns        73842
BM_fma_corobatch<CbType::No, PromPoolAlloc::No, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/8               2074 ns         2074 ns       364253
BM_fma_corobatch<CbType::No, PromPoolAlloc::No, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/64             15568 ns        15543 ns        43683
BM_fma_corobatch<CbType::No, PromPoolAlloc::No, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/512           154197 ns       153620 ns         6384
BM_fma_corobatch<CbType::No, PromPoolAlloc::No, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/4096         1285174 ns      1284676 ns          615
BM_fma_corobatch<CbType::No, PromPoolAlloc::No, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/8192         2451729 ns      2451634 ns          320
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::No, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/8              2650 ns         2650 ns       239802
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::No, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/64            17003 ns        17002 ns        53457
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::No, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/512          109025 ns       109023 ns         7120
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::No, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/4096         846193 ns       846191 ns          769
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::No, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/8192        1572966 ns      1572837 ns          483
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/8             1202 ns         1202 ns       581970
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/64            7713 ns         7713 ns        97508
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/512          63648 ns        63647 ns        11032
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/4096        514935 ns       514331 ns         1606
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No>/8192        849738 ns       849486 ns          806
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::No, RingBuffExec::No>/8            1471 ns         1471 ns       520123
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::No, RingBuffExec::No>/64           6940 ns         6940 ns        96679
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::No, RingBuffExec::No>/512         51924 ns        51923 ns        11909
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::No, RingBuffExec::No>/4096       396500 ns       396479 ns         1838
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::No, RingBuffExec::No>/8192       797192 ns       797190 ns          804
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::Yes, RingBuffExec::No>/8            933 ns          933 ns       725446
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::Yes, RingBuffExec::No>/64          4505 ns         4505 ns       155618
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::Yes, RingBuffExec::No>/512        32884 ns        32841 ns        21041
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::Yes, RingBuffExec::No>/4096      261159 ns       260903 ns         2671
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::Yes, RingBuffExec::No>/8192      520049 ns       519834 ns         1305
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::Yes, RingBuffExec::Yes>/8           744 ns          744 ns       922323
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::Yes, RingBuffExec::Yes>/64         4178 ns         4178 ns       167219
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::Yes, RingBuffExec::Yes>/512       31778 ns        31777 ns        21687
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::Yes, RingBuffExec::Yes>/4096     251526 ns       251521 ns         2757
BM_fma_corobatch<CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::Yes, RingBuffExec::Yes>/8192     507332 ns       507197 ns         1353
```

Out of the box, the overhead the library adds on that machine is roughly `250ns` per task.

Providing a more tailored memory management for the promise allocation and the batch performances, in addition to removing the type erasure introduced by `std::function` can increase performances considerably, bringing the overhead per task to `100ns`.

Configuring the library to allow the execution of the tasks in a single executor, and providing an executor which does not perform dynamic memory allocation, brings the overhead per task to `60ns`.
