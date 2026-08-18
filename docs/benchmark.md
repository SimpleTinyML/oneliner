# Benchmark

This document collects the performance and footprint characteristics of Oneliner across models and targets.

> **Status: Work in progress.** Latency and footprint numbers for the tested models are below. Methodology and reproduction steps are still to be documented.

## Inference Latency

Measured with `oneliner-profiler` under Ariel OS. Latency is the mean wall-clock time of one inference; all runs below report a single sample. On the Raspberry Pi Pico (RP2040), results are given for both the multicore and the sequential scheduler; on the nRF52840DK the scheduler setting is as logged.

### LeNet5 (int8)

| Target | Scheduler | Latency (ms) |
| --- | --- | --- |
| Raspberry Pi Pico | multicore | 32.2 |
| Raspberry Pi Pico | sequential | 43.2 |
| nRF52840DK | — | 42.5 |

### MCUNet visual wake word, 10 fps (int8)

| Target | Scheduler | Latency (ms) |
| --- | --- | --- |
| Raspberry Pi Pico | multicore | 572.2 |
| Raspberry Pi Pico | sequential | 900.5 |
| nRF52840DK | — | 699.8 |

## Footprint

From the `Model artifact sizes` report (bytes): input/output tensor sizes, parameter bytes placed in flash (weights/constants) and arena RAM.

| Model | Input | Output | Params | RAM |
| --- | --- | --- | --- | --- |
| LeNet5 (int8) | 3136 | 40 | 46336 | 5184 |
| MCUNet VWW 10 fps (int8) | 12288 | 2 | 416320 | 106432 |

## Comparison

Preliminary observations from the numbers above:

- On the RP2040, the multicore scheduler cuts inference latency by roughly 25% for LeNet5 (32.2 ms vs 43.2 ms) and 36% for MCUNet VWW (572.2 ms vs 900.5 ms).
- The nRF52840DK lands between the two RP2040 scheduler modes for both models.