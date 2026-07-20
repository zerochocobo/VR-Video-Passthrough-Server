# PT RTX VSR CUDA bridge

This bridge is based on the CUDA implementation shipped in NVIDIA RTX Video SDK 1.1.0's `RTX_Video_API` sample. It exposes the sample C API from a DLL so Python can call VSR with CUDA device pointers.

Build on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File models\rtx_vsr\native\build.ps1
```

The output is written to `models/rtx_vsr/runtime/pt_rtx_vsr_bridge.dll`, next to the release `nvngx_vsr.dll`.
