# Verifying the runtime on another vendor

The dev box is NVIDIA-only. The AMD, Intel and CPU paths are written and can be
exercised on any machine with a vendor override, but only a real AMD or Intel
GPU shows whether the vendor's own backend (ROCm, Vulkan) is picked and how it
performs. This page is for whoever has such a machine.

## What differs per vendor

| | NVIDIA | AMD | Intel | CPU |
|---|---|---|---|---|
| Ollama | inside the distro (CUDA) | native Windows, standalone zip + ROCm zip | native Windows, standalone zip (Vulkan) | native Windows, standalone zip |
| Speaches, ComfyUI, CV server | CUDA containers | CPU containers | CPU containers | CPU containers |
| Spaces | pull the Hub's image | pull, or build locally for the CPU | same | same |
| `hardware.wsl_gpu` | true | false | false | false |

Native Ollama is installed from `ollama-windows-amd64.zip` (plus
`ollama-windows-amd64-rocm.zip` on AMD) into `%LOCALAPPDATA%\ai-app-store\ollama\`
and started headless by the store. A copy installed with `OllamaSetup.exe` is
used instead when present. The interactive installer's silent mode is not
reliable (ollama/ollama#7969), which is why the zip is used.

## Steps on the target machine

1. Install the MSI from the latest CI run or release and open the store.
2. Setup: the Hardware card must show the right vendor, every GPU with VRAM,
   any NPU, and the runtime plan for that vendor. Screenshot it.
3. Click "Install". Watch the log: on non-NVIDIA machines the WSL Ollama step is
   skipped and an "Ollama runs natively on Windows" step downloads the zip.
4. Browse, Models tab: run a small GGUF (any `Q4_K_M` under 5 GB). Chat with
   it. Then note what `ollama ps` reports (the verification script prints it):
   `size_vram` above zero means the GPU backend is in use.
5. Audio: Start, download `faster-whisper-small`, record and transcribe, speak
   a reply. Slow is expected (CPU); failing is not.
6. Canvas: Start, download sd-turbo, generate one 512 x 512 image. Expect about
   a minute on the CPU.
7. Vision: Start the CV server (OpenVINO Model Server on non-NVIDIA), download
   YOLOv10n, load an image, Detect. Expect boxes with scores in under a second.
8. Browse, Apps tab: "Build locally" on a small gradio Space, for example
   `lampongyuen/Gradio-Hello-World-3`. It must reach Running and open.
9. Run the checklist and paste its output into the issue:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\verify-vendor.ps1 -Expect amd
   ```

   The script is read-only. It prints one Markdown checkbox per item in the
   issue's checklist and exits non-zero when something failed.

## Exercising a vendor path without the hardware

Set `AIAS_VENDOR` before starting the store to make it treat the machine as
another vendor:

```powershell
$env:AIAS_VENDOR = "amd"
& "C:\Program Files\AI App Store\ai-app-store-poc.exe"
```

The Hardware card then shows a "forced via AIAS_VENDOR" badge. Everything keyed
off the vendor follows: native Ollama, CPU containers, OpenVINO Model Server,
CPU Space builds. The GPU itself does not change, so on an NVIDIA box the
native Ollama will still use CUDA; the override checks the plumbing, not the
backend. Unset the variable and restart to return to the probed vendor. On an
NVIDIA machine the WSL Ollama is disabled by the forced provisioning run;
re-enable it afterwards with `wsl -d ai-app-store -u root --exec systemctl enable --now ollama`.

## Known limits

- WSL2 exposes no AMD or Intel GPU and no NPU to containers, which is why the
  LLM runs natively and the other services run on the CPU on those vendors.
  GPU-accelerated speech and images on AMD / Intel are tracked in #13.
- Ollama's ROCm backend covers RDNA 2 and newer discrete Radeon cards and the
  Ryzen AI Max integrated GPUs; other AMD parts fall back to Vulkan or the CPU.
- NPUs are detected and shown; nothing uses them yet.
