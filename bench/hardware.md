# Benchmark hardware

All numbers in this repo were measured on this machine unless stated otherwise.

| | |
|---|---|
| Machine | Laptop, Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic |
| CPU | Intel Core i5-1235U: 2 fast cores (CPUs 0-3, up to 4.4 GHz) + 8 slower cores (CPUs 4-11, up to 3.3 GHz) |
| Vector instructions | SSE4.2, AVX2, FMA (no AVX-512) |
| Cache | L1 data 48 KiB per fast core, L2 6.5 MiB total, L3 12 MiB shared |
| RAM | 14 GB |
| Disk | WDC PC SN530 NVMe SSD, 512 GB |
| Compiler | GCC 15.2 |
| Python | 3.14, packages from `bench/requirements.txt` |

## How benchmarks are run

- Single-thread searches run on CPU 2 (a fast core).
- Laptop plugged in, power mode set to Performance.
- Heavy apps closed before large runs.
