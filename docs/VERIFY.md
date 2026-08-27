# Verifying the runtime on another vendor

The dev box is NVIDIA-only. The AMD, Intel and CPU paths are written and can be
exercised on any machine with a vendor override, but only a real AMD or Intel
GPU shows whether the vendor's own backend (ROCm, Vulkan) is picked and how it
performs. This page is for whoever has such a machine.

## What differs per vendor

| | NVIDIA | AMD | Intel | CPU |
|---|---|---|---|---|
| Ollama | inside the distro (CUDA) | native Windows, standalone zip + ROCm zip | native Windows, standalone zip (Vulkan) | native Windows, standalone zip |
| Speech to text | Speaches CUDA container | whisper.cpp native (Vulkan) | whisper.cpp native (Vulkan) | Speaches CPU container |
| Text to speech | Speaches CUDA container | Speaches CPU container | Speaches CPU container | Speaches CPU container |
| Images | ComfyUI CUDA container | ComfyUI portable native (ROCm) | ComfyUI portable native (XPU) | ComfyUI CPU container |
| CV server | Triton container | OpenVINO Model Server container (CPU) | same | same |
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
   any NPU, and the runtime plan for that vendor. The status list has a
   "Virtualization (UEFI)" row: it must be green (hypervisor running, or VT
   enabled in firmware) before provisioning can install WSL2. If it is red,
   the card explains the UEFI toggle; that step is the one thing the store
   cannot do for the user. Screenshot it.
3. Click "Install". Watch the log: on non-NVIDIA machines the WSL Ollama step is
   skipped and an "Ollama runs natively on Windows" step downloads the zip.
4. Browse, Models tab: run a small GGUF (any `Q4_K_M` under 5 GB). Chat with
   it. Then note what `ollama ps` reports (the verification script prints it):
   `size_vram` above zero means the GPU backend is in use.
5. Audio: Start Speaches (TTS). On AMD / Intel also download the whisper.cpp
   model and Start it: the card must show `vulkan` and reach Running; record
   and transcribe, then check with Task Manager (GPU tab, whisper-server.exe)
   that the GPU is used. Speak a reply through Speaches.
6. Canvas: Start. On AMD / Intel the first start downloads and unpacks the
   official ComfyUI portable (1.8 GB); the card shows `rocm` or `xpu`. Download
   sd-turbo, generate one 512 x 512 image and note the time (a CPU run on the
   dev box takes 11 s for 2 steps; the GPU build should be well under that).
   ComfyUI's ROCm build refuses to start on a machine without a supported
   Radeon (RDNA 3 or newer); that is the expected failure on the wrong GPU.
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
  LLM, speech to text and image generation run natively on those vendors and
  only text to speech and the CV server stay in CPU containers.
- The whisper.cpp download comes from this repository's `runtimes` release.
  While the repository is private that URL needs `AIAS_GITHUB_TOKEN` in the
  store's environment (development only); a shipped store downloads from a
  public host, the same constraint the updater has.
- Ollama's ROCm backend covers RDNA 2 and newer discrete Radeon cards and the
  Ryzen AI Max integrated GPUs; other AMD parts fall back to Vulkan or the CPU.
- Native Ollama is started with `OLLAMA_VULKAN=1` and `OLLAMA_IGPU_ENABLE=1`.
  Ollama ships the Vulkan backend but leaves it off by default and drops
  integrated GPUs; without both flags it runs on the CPU. Verified: on an Intel
  Core Ultra 5 225H the Arc 130T iGPU takes 100% of the inference through
  Vulkan (`ollama ps` shows `100% GPU`).
- On an Intel machine with an NPU, the CV server runs as a native Windows OVMS
  process targeting the NPU (`target_device: NPU`); the Vision card shows backend
  `openvino-npu`, runtime native. It is native because WSL2 exposes no NPU to
  containers (only `/dev/dxg`, no `/dev/accel`). Verified on a Core Ultra 5 225H:
  OVMS loads YOLOv10n on the NPU (model state AVAILABLE, target device NPU). This
  is the first NPU-backed path; other modalities still leave the NPU unused.
