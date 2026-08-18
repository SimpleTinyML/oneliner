# Target support

This document tracks the host and embedded targets Oneliner has been verified on, together with the supported feature set for each.

> **Status: Work in progress.** The matrix will be filled in as targets are validated. Sections below are the intended structure.

Host Targets (std) are supported anytime.


## Embedded Targets (no_std)

Oneliner's runtime and executors build on the embedded environments provided by Ariel OS and Embassy. In principle, any target supported by Ariel OS or Embassy can be extended to run Oneliner — with the exception of Xtensa-based targets (e.g. ESP32 series), whose support is currently work in progress.

Real-hardware testing is currently performed on:

- **nRF52840DK** (Cortex-M4) with Ariel OS
- **Raspberry Pi Pico** (RP2040, Cortex-M0+) with Embassy

Most other ARM Cortex-M targets are validated through QEMU simulation, which covers the full compile-to-inference pipeline without physical boards.


## Compatibility Notes

<!-- Known limitations, quirks, and version pinning requirements per target. -->
