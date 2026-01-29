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

CSV output:

```
cd tests
make clean
TMA_BW_CSV=tma_bandwidth.csv make EXTRA_NVCCFLAGS=" -DTEST_THREAD_MEMORY_TILE_TMA_BW" run
```

HBM-focused sweep (larger footprint + L2 evict hint):

```
TMA_BW_FOOTPRINT_MB=2048 TMA_BW_CACHE=evict_first \
  TMA_BW_CSV=tma_bandwidth.csv \
  make EXTRA_NVCCFLAGS=" -DTEST_THREAD_MEMORY_TILE_TMA_BW" run
```

Plot (requires matplotlib):

```
python plot_tma_bandwidth.py --csv tma_bandwidth.csv --out tma_bandwidth.png
```

Optional tunables (compile-time defines):

* `TMA_BW_TILE_H` / `TMA_BW_TILE_W`: tile size in multiples of 16 (default 4x4).
* `TMA_BW_WARPS_PER_BLOCK`: warps per CTA (default 4).
* `TMA_BW_OPS_PER_ITER`: TMA ops per iteration per warp (default 4).
* `TMA_BW_ITERS`: iterations per CTA (default 2000).
* `TMA_BW_WARMUP`: warmup launches before timing (default 2).
* `TMA_BW_REPEATS`: timed launches, best-of used (default 5).
* `TMA_BW_SATURATION`: saturation ratio (default 0.90).

Optional runtime env vars:
* `TMA_BW_CSV`: write CSV to this path.
* `TMA_BW_FOOTPRINT_MB`: override footprint (MB) to reduce L2 reuse.
* `TMA_BW_CACHE`: `normal`, `evict_first`, or `evict_last`.

Note: This test is only compiled on Hopper or Blackwell targets.
