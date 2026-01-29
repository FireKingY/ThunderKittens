# Tests

## TMA bandwidth sweep (TMA ld/st)

This benchmark sweeps the number of active SMs (by launching 1 CTA per SM) and
reports achieved GB/s for TMA load (ld) and TMA store (st). It prints the first
SM count that reaches 90% of the peak measured bandwidth.

Build and run:

```
cd tests
make clean
make EXTRA_NVCCFLAGS=" -DTEST_THREAD_MEMORY_TILE_TMA_BW" run
```

Optional tunables (compile-time defines):

* `TMA_BW_TILE_H` / `TMA_BW_TILE_W`: tile size in multiples of 16 (default 4x4).
* `TMA_BW_WARPS_PER_BLOCK`: warps per CTA (default 4).
* `TMA_BW_OPS_PER_ITER`: TMA ops per iteration per warp (default 4).
* `TMA_BW_ITERS`: iterations per CTA (default 2000).
* `TMA_BW_WARMUP`: warmup launches before timing (default 2).
* `TMA_BW_REPEATS`: timed launches, best-of used (default 5).
* `TMA_BW_SATURATION`: saturation ratio (default 0.90).

Note: This test is only compiled on Hopper or Blackwell targets.
