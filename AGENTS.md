# AGENTS

This repository contains ThunderKittens, a header-only CUDA library plus
standalone kernel implementations and demos.

## Repository layout
- include/: core ThunderKittens headers, types, and primitives.
- kernels/: self-contained kernels with Makefile, benchmarks, and tests.
- tests/: unit tests for ThunderKittens primitives.
- demos/: Python demos that use the kernels.
- assets/: images used in documentation.

## Build and test workflows
- Kernel workflow:
  - cd kernels/<kernel_dir>
  - make
  - make run
  - Run local test_correctness.py or benchmark.py if present.

- Unit tests:
  - cd tests
  - make -j
  - mkdir -p outputs
  - ./unit_tests printout
  - make clean

## Environment notes
- Requires CUDA 12.8+ and a C++20-capable toolchain.
- Primary GPU targets are H100 and B200; other GPUs may be unsupported.

## Agent guidance
- Keep kernel changes localized to their directory and update the local
  Makefile/tests when needed.
- When modifying kernels, compile and run correctness/benchmark tests.
- Review .claude/CLAUDE.md for any additional repo rules.
