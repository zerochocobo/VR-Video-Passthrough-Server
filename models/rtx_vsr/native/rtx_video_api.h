////----------------------------------------------------------------------------------
// File:        rtx_video_api.h
// SDK Version: 1.1.0
//
// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
//----------------------------------------------------------------------------------


///////////////////////////////////////////
// rtx_video_api.h
//
// This provides a simple interface to use the RTX Video SDK.
// There are create, evaluate, shutdown functions for DX11, DX12, CUDA, Vulkan
// The application calls create at startup, shutdown at exit.
// To apply the VSR and/or TrueHDR, call evalulate with input and output textures, and sizes.
//

#pragma once

#if defined(_WIN32)
#include <cstdint>
#else
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wnarrowing"
#ifdef __cplusplus
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif
#endif

#ifdef __cplusplus
#if defined(RTX_VSR_BRIDGE_EXPORTS) && defined(_WIN32)
#define C_API "C" __declspec(dllexport)
#else
#define C_API "C"
#endif
#else
#define C_API
#endif

typedef struct
{
    uint32_t QualityLevel;
} API_VSR_Setting;

typedef struct
{
    uint32_t Contrast;
    uint32_t Saturation;
    uint32_t MiddleGray;
    uint32_t MaxLuminance;
} API_THDR_Setting;

typedef unsigned int API_BOOL;
#define API_BOOL_FAIL       0
#define API_BOOL_SUCCESS    1

typedef struct
{
    uint32_t left;
    uint32_t top;
    uint32_t right;
    uint32_t bottom;
} API_RECT;

#if defined(_WIN32)
#ifdef __cplusplus

extern C_API void pt_rtx_vsr_set_app_path(const wchar_t* appPath);
extern C_API void pt_rtx_vsr_set_paths(const wchar_t* featurePath, const wchar_t* dataPath);

////////////////////////////////////////////////////////////////////
// DX11
struct ID3D11Device;
struct ID3D11Texture2D;

// For DX11, first create ID3D11Device and pass it in.
extern C_API API_BOOL rtx_video_api_dx11_create(ID3D11Device* pD3DDevice, API_BOOL THDREnable, API_BOOL VSREnable);
// Ideally the output surface should be created with D3D11_BIND_UNORDERED_ACCESS, but is not required.
extern C_API API_BOOL rtx_video_api_dx11_evaluate(ID3D11Texture2D* pInput, ID3D11Texture2D* pOutput, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting);
extern C_API void rtx_video_api_dx11_shutdown();

////////////////////////////////////////////////////////////////////
// DX12
struct ID3D12Device;
struct ID3D12Resource;
struct ID3D12Fence;

// For DX12, first create ID3D12Device and pass it in, along with GPU selection.
extern C_API API_BOOL rtx_video_api_dx12_create(ID3D12Device* pD3DDevice, uint32_t uGPUNodeMask, uint32_t uGPUVisibleNodeMask, API_BOOL THDREnable, API_BOOL VSREnable);
// Ideally the output surface should be created with D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, but is not required. Fences will be waited by GPU on and signaled, incrementing value.
extern C_API API_BOOL rtx_video_api_dx12_evaluate(ID3D12Resource* pInput, ID3D12Resource* pOutput, ID3D12Fence* pInFence, uint64_t& qwInFenceValue, ID3D12Fence* pOutFence, uint64_t& qwOutFenceValue,
                                                  API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting);
extern C_API void rtx_video_api_dx12_shutdown();

#endif      // c++
#endif      // win32

////////////////////////////////////////////////////////////////////
// CUDA works on CUarray input/output. If CUdeviceptr or hostptr is used, stride must be 4*width (ie, cannot be FP16 output).
extern C_API API_BOOL rtx_video_api_cuda_create(void* cuContext, void* cuStream, int iGpu, API_BOOL THDREnable, API_BOOL VSREnable);
extern C_API API_BOOL rtx_video_api_cuda_evaluate(uint64_t cuTexObject_Input, uint64_t cuSurfObject_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting);
extern C_API API_BOOL rtx_video_api_cuda_evaluate_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting);
extern C_API API_BOOL rtx_video_api_cuda_evaluate_hostptr(void* hostptr_Input, void* hostptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting);
extern C_API void rtx_video_api_cuda_shutdown();

////////////////////////////////////////////////////////////////////
// VULKAN works on NVSDK_NGX_Resource_VK input/output. 
// pMiddle is needed only if both VSR and THDR are enabled.
// pMiddle must created to match output size in VK_FORMAT_R8G8B8A8_UNORM for 8 bit input and VK_FORMAT_A2B10G10R10_UNORM_PACK32 for 10 bit input.
extern C_API API_BOOL rtx_video_api_vulkan_create(API_BOOL THDREnable, API_BOOL VSREnable, void* hDevice, void* hInstance, void* hPhysicalDevice, void* hCommandBuffer);
extern C_API API_BOOL rtx_video_api_vulkan_evaluate(void* pInput, void* pOutput, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting, void* pMiddle);
extern C_API void rtx_video_api_vulkan_shutdown();
