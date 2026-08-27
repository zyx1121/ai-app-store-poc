# ADR 0001: Accelerator adapters and the device onboarding SOP

Status: accepted

## Context

The store ships pre-installed on many AI PCs, and new models arrive often: Intel
Core Ultra with an Arc iGPU and an NPU, AMD Ryzen AI Max with a Radeon iGPU and
an XDNA NPU, machines with an SSD-backed KV cache (Phison aiDAPTIV+), and
whatever comes next. Each brings a different accelerator and a different way to
reach it (CUDA, ROCm, Vulkan, Intel XPU / oneAPI, OpenVINO on the NPU, a vendor
middleware). We need to add support for a new machine quickly and to a fixed
recipe, without reworking the core each time.

Two facts from bringing up real hardware bound the design:

- The unified surface is already the API, not the binary. Every base UI talks to
  a fixed `localhost` endpoint (OpenAI-compatible for LLM, KServe v2 for CV, an
  OpenAI-style speech API), and behind it the store swaps the runtime per device.
  See [ARCHITECTURE.md](../ARCHITECTURE.md).
- Real acceleration only appears on the metal. On an Intel Core Ultra 5 225H the
  LLM ran on the CPU until `OLLAMA_VULKAN=1` and `OLLAMA_IGPU_ENABLE=1` were set,
  and on a stale-WSL machine provisioning stalled on an interactive prompt.
  Neither is visible without running on the device.

## Decision

Model per-device support as a set of **accelerator adapters** behind the fixed
API, resolved from a host probe, driven by one lifecycle. Keep the core free of
per-vendor knowledge; put that knowledge in adapters and in a written SOP.

### What stays in the core (unchanged per device)

- The host probe (`hardware.ps1` -> `HardwareProfile`): GPUs, NPUs, VRAM,
  firmware virtualization.
- The fixed API contract and ports in `src/lib/api.ts`. Base UIs never change.
- The service lifecycle in `services.rs`: fetch, start, health-check, stop, for
  both runtimes (a container in the WSL2 distro, or a native Windows process).
- The compatibility verdict machinery (`ready` / `maybe` / `incompatible`).

### The adapter seam

Today `spec(id, vendor)` selects a `ServiceSpec` by vendor in a match. Generalise
that selection into an adapter registry. Conceptually each adapter is:

```rust
trait Accelerator {
    /// Is this accelerator present and usable on this host? Cheap, from the probe
    /// plus any device-id / registry / SDK checks the accelerator needs.
    fn detect(host: &HardwareProfile) -> Option<Self> where Self: Sized;

    /// Human label and backend tag shown in Setup (e.g. "vulkan", "rocm", "aidaptiv").
    fn backend(&self) -> &'static str;

    /// The runtime(s) this accelerator provides for each modality, as ServiceSpecs
    /// (Container or Native) that sit behind the fixed localhost API.
    fn services(&self) -> Vec<ServiceSpec>;

    /// Extra env / launch tweaks the runtime needs to actually use the silicon
    /// (the OLLAMA_VULKAN + OLLAMA_IGPU_ENABLE class of thing).
    fn tuning(&self) -> &[(&'static str, &'static str)];

    /// Does a given model fit here? Uses unified memory, not just a VRAM register.
    fn compat(&self, model: &ModelInfo) -> Compat;

    /// One smoke test per modality that proves the silicon is actually used
    /// (e.g. `ollama ps` shows GPU, not CPU), not just that a reply came back.
    fn smoke_tests(&self) -> Vec<SmokeTest>;
}
```

A device is then the set of adapters that `detect()` returns true on: a base GPU
adapter (nvidia / amd / intel / cpu) plus optional capability adapters (an NPU
adapter, an aiDAPTIV adapter). Adapters compose; the LLM adapter for a machine
with aiDAPTIV is the aiDAPTIV one instead of plain Ollama, chosen by priority.

### Modality to backend, today and next

| Modality | NVIDIA | AMD | Intel | CPU | Capability adapters |
|----------|--------|-----|-------|-----|---------------------|
| LLM | Ollama CUDA (WSL) | Ollama ROCm (native) | Ollama Vulkan (native) | Ollama CPU | aiDAPTIV: SSD-backed KV cache for models past memory (gpt-oss-120b-Q4) |
| Detection / CV | Triton | OpenVINO Model Server | OpenVINO Model Server | OpenVINO Model Server | Intel NPU: OVMS target device `NPU` |
| Speech (STT) | Speaches CUDA | whisper.cpp Vulkan (native) | whisper.cpp Vulkan (native) | Speaches CPU | |
| Images | ComfyUI CUDA | ComfyUI ROCm portable (native) | ComfyUI XPU portable (native) | ComfyUI CPU | |

The base GPU rows exist today. The right column is where new adapters land.

## What cannot be modularised (the honest boundary)

Three parts are irreducibly per-vendor engineering. The adapter interface
standardises their shape, not their content.

1. **Detection.** "Has an aiDAPTIV SSD", "is the XDNA NPU usable" are device-id,
   registry-key, or SDK-presence checks with no generic form. Each capability
   needs its own `detect()`.
2. **Runtime glue.** aiDAPTIVLink has its own API and config; an Intel NPU goes
   through OpenVINO while an AMD NPU goes through the VitisAI execution provider.
   Launch, config, and model format differ per backend.
3. **Real-hardware verification.** The CPU-fallback and interactive-prompt bugs
   above only appeared on the metal. New silicon brings driver surprises. The
   SOP therefore always ends on the device, and a smoke test must prove the
   accelerator is used, not merely that a request succeeded.

The goal is not zero-effort auto-support. It is to take a new AI PC from days to
about half a day, on a fixed script that never skips a verification.

## Device onboarding SOP

For each new AI PC:

1. **Probe.** Install the MSI, open Setup, read the detected vendor, GPUs, NPU,
   unified memory, and the virtualization row.
2. **Known accelerator: no code.** If every detected accelerator already has an
   adapter, provision and go to step 5.
3. **New accelerator: one adapter.** Implement `Accelerator` for it: `detect`,
   the `ServiceSpec`(s), `tuning`, `compat`, and `smoke_tests`. Register it. Touch
   no core files.
4. **Provision.** Run Install runtime. It must complete unattended (dism +
   `wsl --update`, native runtime downloads). The only allowed manual step is a
   UEFI virtualization toggle, which the store detects and explains.
5. **Verify on the metal.** Run `scripts/verify-vendor.ps1` and each adapter's
   smoke tests. Every modality must show the accelerator in use (for LLM,
   `ollama ps` shows GPU; for CV, the OVMS/Triton device; and so on), not a
   silent CPU fallback.
6. **Record.** Note the machine, the backend each modality resolved to, and the
   measured numbers (TTFT, tokens/s, image time) in the issue, as done for the
   Intel Core Ultra bring-up.

## Consequences

- Adding a machine that matches known accelerators is declarative and needs no
  core change; the current per-vendor `ServiceSpec` table is the first instance
  of this pattern.
- A genuinely new accelerator is one self-contained adapter plus its smoke tests,
  not edits spread across the codebase.
- Verification stays mandatory and on-device. Silent CPU fallback is treated as a
  failure, caught by the smoke test, not by "a reply came back".
- Migrating the current `spec(id, vendor)` match to the registry is the first
  implementation step; the aiDAPTIV LLM adapter and the Intel NPU CV adapter are
  the first two capability adapters to write against it.
