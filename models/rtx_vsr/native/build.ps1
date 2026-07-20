param(
    # RTX Video SDK 1.1.0 was validated with the CUDA 12.x runtime on the
    # target RTX hardware.  CUDA 13 builds can fail during NGX initialization.
    [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$rtxRoot = Split-Path -Parent $root
$vsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
$cl = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64\cl.exe"

if (-not (Test-Path -LiteralPath $vsDevCmd)) { throw "VsDevCmd.bat not found: $vsDevCmd" }
if (-not (Test-Path -LiteralPath $cl)) { throw "cl.exe not found: $cl" }
if (-not (Test-Path -LiteralPath (Join-Path $CudaRoot "include\cuda.h"))) { throw "CUDA headers not found under $CudaRoot" }

$outDir = Join-Path $rtxRoot "runtime"
$objDir = Join-Path $root "build"
New-Item -ItemType Directory -Force $outDir, $objDir | Out-Null

$cmd = @"
call "$vsDevCmd" -arch=x64 -host_arch=x64 -vcvars_ver=14.29 >nul &&
"$cl" /nologo /std:c++17 /O2 /EHsc /MT /DRTX_VSR_BRIDGE_EXPORTS /LD
 /I"$root" /I"$root\sdk\include" /I"$CudaRoot\include"
 "$root\rtx_video_api_cuda_impl.cpp"
 /Fo"$objDir\rtx_video_api_cuda_impl.obj"
 /link /LIBPATH:"$root\sdk\lib" /LIBPATH:"$CudaRoot\lib\x64"
 /OUT:"$outDir\pt_rtx_vsr_bridge.dll" /IMPLIB:"$objDir\pt_rtx_vsr_bridge.lib"
"@
$cmd = ($cmd -replace "`r?`n", " ")
& cmd.exe /d /s /c $cmd
if ($LASTEXITCODE -ne 0) { throw "RTX VSR bridge build failed with exit code $LASTEXITCODE" }

$cudaRuntime = Get-ChildItem (Join-Path $CudaRoot "bin") -Filter "cudart64_*.dll" | Select-Object -First 1
if (-not $cudaRuntime) { throw "CUDA runtime DLL not found under $CudaRoot\bin" }
Copy-Item -Force $cudaRuntime.FullName (Join-Path $outDir $cudaRuntime.Name)

Write-Host "Built $outDir\pt_rtx_vsr_bridge.dll"
