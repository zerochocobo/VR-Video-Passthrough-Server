//----------------------------------------------------------------------------------
// File:        rtx_video_api_cuda_impl.cpp
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

/**
*  This sample application demonstrates use of RTX Video SDK
*  by providing an api taking CUDA input and output.
*  Inputs for CUDA are ARGB/ABGR10.
*  Output from VSR is in ARGB or if input is 10 bit, ABGR10.
*  Output from THDR is in 10 bit ABGR10.
*  If both are enabled then VSR -> THDR.
*/

#include "rtx_video_api.h"

#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_helpers_truehdr.h>
#include <nvsdk_ngx_helpers_vsr.h>

// link with nvsdk_ngx_<X>.lib

#include <cuda.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>

#include "utils.h"

#if defined(_WIN32)
#pragma comment(lib, "nvsdk_ngx_s.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#endif

#define CUDA_VERSION_INT_101010_2_DEFINED           12080       // this is based on cuda.h 

static std::wstring g_pt_rtx_vsr_app_path = L".";
static std::wstring g_pt_rtx_vsr_data_path = L".";

const wchar_t* pt_rtx_vsr_app_path()
{
    return g_pt_rtx_vsr_app_path.c_str();
}

const wchar_t* pt_rtx_vsr_data_path()
{
    return g_pt_rtx_vsr_data_path.c_str();
}

extern "C" __declspec(dllexport) void pt_rtx_vsr_set_app_path(const wchar_t* appPath)
{
    g_pt_rtx_vsr_app_path = (appPath && appPath[0]) ? appPath : L".";
}

extern "C" __declspec(dllexport) void pt_rtx_vsr_set_paths(const wchar_t* featurePath, const wchar_t* dataPath)
{
    g_pt_rtx_vsr_app_path = (featurePath && featurePath[0]) ? featurePath : L".";
    g_pt_rtx_vsr_data_path = (dataPath && dataPath[0]) ? dataPath : L".";
}

#if (CUDA_VERSION < CUDA_VERSION_INT_101010_2_DEFINED)
  #define CU_AD_FORMAT_UNORM_INT_101010_2 ((CUarray_format)0x50)
#endif

#if defined(_WIN32)
#define NGX_BREAKPOINT() __debugbreak()
#else
#include <signal.h>
#define NGX_BREAKPOINT() raise(SIGTRAP)
#endif

#define CHECK_NGX(func)                                                                             \
{                                                                                                   \
    NVSDK_NGX_Result status = (func);                                                               \
    if(status != NVSDK_NGX_Result_Success) {                                                        \
        std::cerr << "NGX error at : " << __FILE__ << ":" << __LINE__ << " : "                      \
        << status << std::endl;                                                                     \
        return API_BOOL_FAIL;                                                                        \
        }                                                                                           \
}

#define CUDADRV_CHECK(x)                                                                            \
{                                                                                                   \
    CUresult rval;                                                                                  \
    if ((rval = (x)) != CUDA_SUCCESS)                                                               \
    {                                                                                               \
        const char *error_str;                                                                      \
        cuGetErrorString(rval, &error_str);                                                         \
        std::printf("%s():%i: CUDA driver API error: \"%s\"\n", __FUNCTION__, __LINE__, error_str); \
        return API_BOOL_FAIL;                                                                       \
    }                                                                                               \
}


class cuda_api_impl
{
    NVSDK_NGX_Parameter*        m_ngxParameters             = nullptr;
    NVSDK_NGX_Handle*           m_TrueHDRFeature            = nullptr;
    NVSDK_NGX_Handle*           m_VSRFeature                = nullptr;

    CUdevice                    m_cuDevice                  = 0;
    CUcontext                   m_cuContext                 = NULL;

    bool                        m_bNeedMiddle               = false;

    CUarray                     m_cuArrayMid                = nullptr;
    CUtexObject                 m_cuTexObjectMid            = 0;
    CUsurfObject                m_cuSurfObjectMid           = 0;
    size_t                      m_uMiddleWidth              = 0;
    size_t                      m_uMiddleHeight             = 0;

    CUarray                     m_cuArraySrc                = nullptr;
    CUtexObject                 m_cuTexObjectSrc            = 0;
    size_t                      m_uSrcArrayWidth            = 0;
    size_t                      m_uSrcArrayHeight           = 0;

    CUarray                     m_cuArrayDst                = nullptr;
    CUsurfObject                m_cuSurfObjectDst           = 0;
    size_t                      m_uDstArrayWidth            = 0;
    size_t                      m_uDstArrayHeight           = 0;


public:
    API_BOOL create(void* cuContext, void* cuStream, int iGpu, API_BOOL THDREnable, API_BOOL VSREnable);
    API_BOOL evaluate(uint64_t cuTexObject_Input, uint64_t cuSurfObject_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting);
    API_BOOL evaluate_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting);
    API_BOOL evaluate_hostptr(void* hostptr_Input, void* hostptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting);
    void shutdown();
};

API_BOOL cuda_api_impl::create(void* cuContext, void* cuStream, int iGpu, API_BOOL THDREnable, API_BOOL VSREnable)
{
    std::fprintf(stderr, "[pt-rtx-vsr] create begin gpu=%d\n", iGpu);
    std::fflush(stderr);
    if (!cuContext)
    {
        CUDADRV_CHECK(cuInit(0));
        std::fprintf(stderr, "[pt-rtx-vsr] cuInit ok\n"); std::fflush(stderr);
        CUDADRV_CHECK(cuDeviceGet(&m_cuDevice, iGpu));
        std::fprintf(stderr, "[pt-rtx-vsr] cuDeviceGet ok\n"); std::fflush(stderr);
        CUDADRV_CHECK(cuDevicePrimaryCtxRetain(&m_cuContext, m_cuDevice));
        std::fprintf(stderr, "[pt-rtx-vsr] primary context ok\n"); std::fflush(stderr);
        CUDADRV_CHECK(cuCtxSetCurrent(m_cuContext));
        std::fprintf(stderr, "[pt-rtx-vsr] current context set\n"); std::fflush(stderr);
        cuContext = m_cuContext;
    }

    const wchar_t* featurePaths[] = { APP_PATH };
    NVSDK_NGX_FeatureCommonInfo featureInfo = {};
    featureInfo.PathListInfo.Path = featurePaths;
    featureInfo.PathListInfo.Length = 1;
    std::fprintf(stderr, "[pt-rtx-vsr] NGX init feature=%ls data=%ls\n", APP_PATH, pt_rtx_vsr_data_path()); std::fflush(stderr);
    CHECK_NGX(NVSDK_NGX_CUDA_Init(APP_ID, pt_rtx_vsr_data_path(), &featureInfo));
    std::fprintf(stderr, "[pt-rtx-vsr] NGX init ok\n"); std::fflush(stderr);

    CHECK_NGX(NVSDK_NGX_CUDA_GetCapabilityParameters(&m_ngxParameters));
    std::fprintf(stderr, "[pt-rtx-vsr] capability params ok\n"); std::fflush(stderr);

    m_bNeedMiddle = (THDREnable && VSREnable);

    if (THDREnable)
    {
        int TrueHDRAvailable = 0;
        CHECK_NGX(m_ngxParameters->Get(NVSDK_NGX_Parameter_TrueHDR_Available, &TrueHDRAvailable));
        if (!TrueHDRAvailable) return API_BOOL_FAIL;

        // check ScratchBufferSize - truehdr is not expected to request any
        size_t byteSize = 0;
        CHECK_NGX(NVSDK_NGX_CUDA_GetScratchBufferSize(NVSDK_NGX_Feature_TrueHDR, m_ngxParameters, &byteSize));
        if (byteSize != 0) return API_BOOL_FAIL;

        NVSDK_NGX_CUDA_TRUEHDR_Create_Params TrueHDRCreateParams = {};
        TrueHDRCreateParams.InCUContext = cuContext;
        TrueHDRCreateParams.InCUStream = cuStream;
        CHECK_NGX(NGX_CUDA_CREATE_TRUEHDR(&m_TrueHDRFeature, m_ngxParameters, &TrueHDRCreateParams));
    }
    if (VSREnable)
    {
        int VSRAvailable = 0;
        CHECK_NGX(m_ngxParameters->Get(NVSDK_NGX_Parameter_VSR_Available, &VSRAvailable));
        if (!VSRAvailable) return API_BOOL_FAIL;

        // check ScratchBufferSize - vsr is not expected to request any
        size_t byteSize = 0;
        CHECK_NGX(NVSDK_NGX_CUDA_GetScratchBufferSize(NVSDK_NGX_Feature_VSR, m_ngxParameters, &byteSize));
        if (byteSize != 0) return API_BOOL_FAIL;

        NVSDK_NGX_CUDA_VSR_Create_Params VSRCreateParams = {};
        VSRCreateParams.InCUContext = cuContext;
        VSRCreateParams.InCUStream = cuStream;
        CHECK_NGX(NGX_CUDA_CREATE_VSR(&m_VSRFeature, m_ngxParameters, &VSRCreateParams));
    }

    return API_BOOL_SUCCESS;
}

API_BOOL cuda_api_impl::evaluate(uint64_t cuTexObject_Input, uint64_t cuSurfObject_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    if (m_bNeedMiddle && (!m_cuArrayMid || outputRect.right != m_uMiddleWidth || outputRect.bottom != m_uMiddleHeight))
    {
        if (m_cuArrayMid)
        {
            CUDADRV_CHECK(cuArrayDestroy(m_cuArrayMid));
            CUDADRV_CHECK(cuTexObjectDestroy(m_cuTexObjectMid));
            CUDADRV_CHECK(cuSurfObjectDestroy(m_cuSurfObjectMid));
        }

        m_uMiddleWidth     = outputRect.right;
        m_uMiddleHeight    = outputRect.bottom;

        CUarray_format cuArrayFormat = CU_AD_FORMAT_UNSIGNED_INT8;
        CUDA_ARRAY_DESCRIPTOR cuArrayOutputDesc{
            static_cast<size_t>(outputRect.right),
            static_cast<size_t>(outputRect.bottom),
            cuArrayFormat,
            4
        };
        CUDADRV_CHECK(cuArrayCreate(&m_cuArrayMid, &cuArrayOutputDesc));
        {
            CUDA_RESOURCE_DESC resDescOutput;
            memset(&resDescOutput, 0, sizeof(CUDA_RESOURCE_DESC));
            resDescOutput.resType = CU_RESOURCE_TYPE_ARRAY;
            resDescOutput.res.array.hArray = m_cuArrayMid;

            CUDA_TEXTURE_DESC texDescOutput;
            memset(&texDescOutput, 0, sizeof(CUDA_TEXTURE_DESC));
            texDescOutput.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texDescOutput.flags = CU_TRSF_NORMALIZED_COORDINATES;

            CUDADRV_CHECK(cuTexObjectCreate(&m_cuTexObjectMid, &resDescOutput, &texDescOutput, nullptr));
            CUDADRV_CHECK(cuSurfObjectCreate(&m_cuSurfObjectMid, &resDescOutput));
        }
    }


    if (m_VSRFeature)
    {
        NVSDK_NGX_CUDA_VSR_Eval_Params CUDAVsrEvalParams = {};
        CUDAVsrEvalParams.pInput                   = &cuTexObject_Input;
        CUDAVsrEvalParams.pOutput                  = m_bNeedMiddle ? &m_cuTexObjectMid : (CUsurfObject*)&cuSurfObject_Output;
        CUDAVsrEvalParams.InputSubrectBase.X       = inputRect.left;
        CUDAVsrEvalParams.InputSubrectBase.Y       = inputRect.top;
        CUDAVsrEvalParams.InputSubrectSize.Width   = inputRect.right - inputRect.left;
        CUDAVsrEvalParams.InputSubrectSize.Height  = inputRect.bottom - inputRect.top;
        CUDAVsrEvalParams.OutputSubrectBase.X      = outputRect.left;
        CUDAVsrEvalParams.OutputSubrectBase.Y      = outputRect.top;
        CUDAVsrEvalParams.OutputSubrectSize.Width  = outputRect.right - outputRect.left;
        CUDAVsrEvalParams.OutputSubrectSize.Height = outputRect.bottom - outputRect.top;
        CUDAVsrEvalParams.QualityLevel             = (NVSDK_NGX_VSR_QualityLevel)pVSRSetting->QualityLevel;

        CHECK_NGX(NGX_CUDA_EVALUATE_VSR(m_VSRFeature, m_ngxParameters, &CUDAVsrEvalParams));
    }
    if (m_TrueHDRFeature)
    {
        NVSDK_NGX_CUDA_TRUEHDR_Eval_Params CUDATrueHDREvalParams = {};
        CUDATrueHDREvalParams.pInput                   = m_bNeedMiddle ? &m_cuTexObjectMid : (CUtexObject*)&cuTexObject_Input;
        CUDATrueHDREvalParams.pOutput                  = &cuSurfObject_Output;
        CUDATrueHDREvalParams.InputSubrectTL.X         = m_bNeedMiddle ? outputRect.left   : inputRect.left;
        CUDATrueHDREvalParams.InputSubrectTL.Y         = m_bNeedMiddle ? outputRect.top    : inputRect.top;
        CUDATrueHDREvalParams.InputSubrectBR.Width     = m_bNeedMiddle ? outputRect.right  : inputRect.right;
        CUDATrueHDREvalParams.InputSubrectBR.Height    = m_bNeedMiddle ? outputRect.bottom : inputRect.bottom;
        CUDATrueHDREvalParams.OutputSubrectTL.X        = outputRect.left;
        CUDATrueHDREvalParams.OutputSubrectTL.Y        = outputRect.top;
        CUDATrueHDREvalParams.OutputSubrectBR.Width    = outputRect.right;
        CUDATrueHDREvalParams.OutputSubrectBR.Height   = outputRect.bottom;
        CUDATrueHDREvalParams.Contrast                 = pTHDRSetting->Contrast;
        CUDATrueHDREvalParams.Saturation               = pTHDRSetting->Saturation;
        CUDATrueHDREvalParams.MiddleGray               = pTHDRSetting->MiddleGray;
        CUDATrueHDREvalParams.MaxLuminance             = pTHDRSetting->MaxLuminance;

        CHECK_NGX(NGX_CUDA_EVALUATE_TRUEHDR(m_TrueHDRFeature, m_ngxParameters, &CUDATrueHDREvalParams));
    }

    return API_BOOL_SUCCESS;
}


API_BOOL cuda_api_impl::evaluate_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    using timing_clock = std::chrono::steady_clock;
    static uint64_t timing_count = 0;
    static double timing_input_copy_ms = 0.0;
    static double timing_evaluate_ms = 0.0;
    static double timing_output_copy_ms = 0.0;
    const char* timing_env = std::getenv("PT_RTX_VSR_NATIVE_TIMING");
    const bool timing_enabled = timing_env && timing_env[0] == '1';
    if (!m_cuArraySrc || inputRect.right != m_uSrcArrayWidth || inputRect.bottom != m_uSrcArrayHeight)
    {
        if (m_cuArraySrc)
        {
            CUDADRV_CHECK(cuArrayDestroy(m_cuArraySrc));
            CUDADRV_CHECK(cuTexObjectDestroy(m_cuTexObjectSrc));
        }

        m_uSrcArrayWidth     = inputRect.right;
        m_uSrcArrayHeight    = inputRect.bottom;

        CUarray_format cuArrayFormat = CU_AD_FORMAT_UNSIGNED_INT8;
        CUDA_ARRAY_DESCRIPTOR cuArrayOutputDesc{
            m_uSrcArrayWidth,
            m_uSrcArrayHeight,
            cuArrayFormat,
            4
        };
        CUDADRV_CHECK(cuArrayCreate(&m_cuArraySrc, &cuArrayOutputDesc));
        {
            CUDA_RESOURCE_DESC resDescOutput;
            memset(&resDescOutput, 0, sizeof(CUDA_RESOURCE_DESC));
            resDescOutput.resType = CU_RESOURCE_TYPE_ARRAY;
            resDescOutput.res.array.hArray = m_cuArraySrc;

            CUDA_TEXTURE_DESC texDescOutput;
            memset(&texDescOutput, 0, sizeof(CUDA_TEXTURE_DESC));
            texDescOutput.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texDescOutput.flags = CU_TRSF_NORMALIZED_COORDINATES;

            CUDADRV_CHECK(cuTexObjectCreate(&m_cuTexObjectSrc, &resDescOutput, &texDescOutput, nullptr));
        }
    }
    if (!m_cuArrayDst || outputRect.right != m_uDstArrayWidth || outputRect.bottom != m_uDstArrayHeight)
    {
        if (m_cuArrayDst)
        {
            CUDADRV_CHECK(cuArrayDestroy(m_cuArrayDst));
            CUDADRV_CHECK(cuTexObjectDestroy(m_cuSurfObjectDst));
        }

        m_uDstArrayWidth     = outputRect.right;
        m_uDstArrayHeight    = outputRect.bottom;

        CUarray_format cuArrayFormat = m_TrueHDRFeature ? CU_AD_FORMAT_UNORM_INT_101010_2 : CU_AD_FORMAT_UNSIGNED_INT8;
        CUDA_ARRAY_DESCRIPTOR cuArrayOutputDesc{
            m_uDstArrayWidth,
            m_uDstArrayHeight,
            cuArrayFormat,
            4
        };
        CUDADRV_CHECK(cuArrayCreate(&m_cuArrayDst, &cuArrayOutputDesc));
        {
            CUDA_RESOURCE_DESC resDescOutput;
            memset(&resDescOutput, 0, sizeof(CUDA_RESOURCE_DESC));
            resDescOutput.resType = CU_RESOURCE_TYPE_ARRAY;
            resDescOutput.res.array.hArray = m_cuArrayDst;

            CUDADRV_CHECK(cuSurfObjectCreate(&m_cuSurfObjectDst, &resDescOutput));
        }
    }
    const auto timing_input_begin = timing_enabled ? timing_clock::now() : timing_clock::time_point{};
    {
        // copy input from device to array
        CUDA_MEMCPY2D copyParam     = {};
        copyParam.dstMemoryType     = CU_MEMORYTYPE_ARRAY;
        copyParam.dstArray          = m_cuArraySrc;
        copyParam.srcMemoryType     = CU_MEMORYTYPE_DEVICE;
        copyParam.srcDevice         = (CUdeviceptr)cuDeviceptr_Input;
        copyParam.srcPitch          = m_uSrcArrayWidth * 4;
        copyParam.WidthInBytes      = m_uSrcArrayWidth * 4;
        copyParam.Height            = m_uSrcArrayHeight;
        CUDADRV_CHECK(cuMemcpy2D(&copyParam));
    }
    if (timing_enabled) CUDADRV_CHECK(cuCtxSynchronize());
    const auto timing_input_end = timing_enabled ? timing_clock::now() : timing_clock::time_point{};



    API_BOOL res = evaluate(m_cuTexObjectSrc, m_cuSurfObjectDst, inputRect, outputRect, pVSRSetting, pTHDRSetting);
    if (timing_enabled) CUDADRV_CHECK(cuCtxSynchronize());
    const auto timing_evaluate_end = timing_enabled ? timing_clock::now() : timing_clock::time_point{};
;
    if (res == API_BOOL_SUCCESS)
    {
        // copy output from array to device
        CUDA_MEMCPY2D copyParam     = {};
        copyParam.dstMemoryType     = CU_MEMORYTYPE_DEVICE;
        copyParam.dstDevice         = (CUdeviceptr)cuDeviceptr_Output;
        copyParam.dstPitch          = m_uDstArrayWidth * 4;
        copyParam.srcMemoryType     = CU_MEMORYTYPE_ARRAY;
        copyParam.srcArray          = m_cuArrayDst;
        copyParam.WidthInBytes      = m_uDstArrayWidth * 4;
        copyParam.Height            = m_uDstArrayHeight;
        CUDADRV_CHECK(cuMemcpy2D(&copyParam));
    }
    if (timing_enabled) CUDADRV_CHECK(cuCtxSynchronize());

    if (timing_enabled)
    {
        const auto timing_output_end = timing_clock::now();
        const auto elapsed_ms = [](timing_clock::time_point begin, timing_clock::time_point end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        ++timing_count;
        timing_input_copy_ms += elapsed_ms(timing_input_begin, timing_input_end);
        timing_evaluate_ms += elapsed_ms(timing_input_end, timing_evaluate_end);
        timing_output_copy_ms += elapsed_ms(timing_evaluate_end, timing_output_end);
        if (timing_count % 100 == 0)
        {
            std::fprintf(
                stderr,
                "[pt-rtx-vsr] native_avg_ms calls=%llu input_copy=%.3f evaluate=%.3f output_copy=%.3f total=%.3f\n",
                static_cast<unsigned long long>(timing_count),
                timing_input_copy_ms / timing_count,
                timing_evaluate_ms / timing_count,
                timing_output_copy_ms / timing_count,
                (timing_input_copy_ms + timing_evaluate_ms + timing_output_copy_ms) / timing_count
            );
            std::fflush(stderr);
        }
    }

    return res;
}


API_BOOL cuda_api_impl::evaluate_hostptr(void* hostptr_Input, void* hostptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    if (!m_cuArraySrc || inputRect.right != m_uSrcArrayWidth || inputRect.bottom != m_uSrcArrayHeight)
    {
        if (m_cuArraySrc)
        {
            CUDADRV_CHECK(cuArrayDestroy(m_cuArraySrc));
            CUDADRV_CHECK(cuTexObjectDestroy(m_cuTexObjectSrc));
        }

        m_uSrcArrayWidth     = inputRect.right;
        m_uSrcArrayHeight    = inputRect.bottom;

        CUarray_format cuArrayFormat = CU_AD_FORMAT_UNSIGNED_INT8;
        CUDA_ARRAY_DESCRIPTOR cuArrayOutputDesc{
            m_uSrcArrayWidth,
            m_uSrcArrayHeight,
            cuArrayFormat,
            4
        };
        CUDADRV_CHECK(cuArrayCreate(&m_cuArraySrc, &cuArrayOutputDesc));
        {
            CUDA_RESOURCE_DESC resDescOutput;
            memset(&resDescOutput, 0, sizeof(CUDA_RESOURCE_DESC));
            resDescOutput.resType = CU_RESOURCE_TYPE_ARRAY;
            resDescOutput.res.array.hArray = m_cuArraySrc;

            CUDA_TEXTURE_DESC texDescOutput;
            memset(&texDescOutput, 0, sizeof(CUDA_TEXTURE_DESC));
            texDescOutput.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
            texDescOutput.filterMode = CU_TR_FILTER_MODE_LINEAR;
            texDescOutput.flags = CU_TRSF_NORMALIZED_COORDINATES;

            CUDADRV_CHECK(cuTexObjectCreate(&m_cuTexObjectSrc, &resDescOutput, &texDescOutput, nullptr));
        }
    }
    if (!m_cuArrayDst || outputRect.right != m_uDstArrayWidth || outputRect.bottom != m_uDstArrayHeight)
    {
        if (m_cuArrayDst)
        {
            CUDADRV_CHECK(cuArrayDestroy(m_cuArrayDst));
            CUDADRV_CHECK(cuTexObjectDestroy(m_cuSurfObjectDst));
        }

        m_uDstArrayWidth     = outputRect.right;
        m_uDstArrayHeight    = outputRect.bottom;

        CUarray_format cuArrayFormat = m_TrueHDRFeature ? CU_AD_FORMAT_UNORM_INT_101010_2 : CU_AD_FORMAT_UNSIGNED_INT8;
        CUDA_ARRAY_DESCRIPTOR cuArrayOutputDesc{
            m_uDstArrayWidth,
            m_uDstArrayHeight,
            cuArrayFormat,
            4
        };
        CUDADRV_CHECK(cuArrayCreate(&m_cuArrayDst, &cuArrayOutputDesc));
        {
            CUDA_RESOURCE_DESC resDescOutput;
            memset(&resDescOutput, 0, sizeof(CUDA_RESOURCE_DESC));
            resDescOutput.resType = CU_RESOURCE_TYPE_ARRAY;
            resDescOutput.res.array.hArray = m_cuArrayDst;

            CUDADRV_CHECK(cuSurfObjectCreate(&m_cuSurfObjectDst, &resDescOutput));
        }
    }

    {
        // copy input from host to array
        CUDA_MEMCPY2D copyParam     = {};
        copyParam.dstMemoryType     = CU_MEMORYTYPE_ARRAY;
        copyParam.dstArray          = m_cuArraySrc;
        copyParam.srcMemoryType     = CU_MEMORYTYPE_HOST;
        copyParam.srcHost           = hostptr_Input;
        copyParam.srcPitch          = m_uSrcArrayWidth * 4;
        copyParam.WidthInBytes      = m_uSrcArrayWidth * 4;
        copyParam.Height            = m_uSrcArrayHeight;
        CUDADRV_CHECK(cuMemcpy2D(&copyParam));
    }


    API_BOOL res = evaluate(m_cuTexObjectSrc, m_cuSurfObjectDst, inputRect, outputRect, pVSRSetting, pTHDRSetting);
;
    if (res == API_BOOL_SUCCESS)
    {
        // copy output from array to host
        CUDA_MEMCPY2D copyParam     = {};
        copyParam.dstMemoryType     = CU_MEMORYTYPE_HOST;
        copyParam.dstHost           = hostptr_Output;
        copyParam.dstPitch          = m_uDstArrayWidth * 4;
        copyParam.srcMemoryType     = CU_MEMORYTYPE_ARRAY;
        copyParam.srcArray          = m_cuArrayDst;
        copyParam.WidthInBytes      = m_uDstArrayWidth * 4;
        copyParam.Height            = m_uDstArrayHeight;
        CUDADRV_CHECK(cuMemcpy2D(&copyParam));
    }

    return res;
}



void cuda_api_impl::shutdown()
{
    if (m_TrueHDRFeature)
    {
        NVSDK_NGX_CUDA_ReleaseFeature(m_TrueHDRFeature);
    }
    if (m_VSRFeature)
    {
        NVSDK_NGX_CUDA_ReleaseFeature(m_VSRFeature);
    }
    if (m_cuContext)
    {
        cuDevicePrimaryCtxRelease(m_cuDevice);
    }
    if (m_cuArrayMid)
    {
        cuTexObjectDestroy(m_cuTexObjectMid);
        cuSurfObjectDestroy(m_cuSurfObjectMid);
        cuArrayDestroy(m_cuArrayMid);
    }
    if (m_cuArraySrc)
    {
        cuTexObjectDestroy(m_cuTexObjectSrc);
        cuArrayDestroy(m_cuArraySrc);
    }
    if (m_cuArrayDst)
    {
        cuSurfObjectDestroy(m_cuSurfObjectDst);
        cuArrayDestroy(m_cuArrayDst);
    }

    if (m_ngxParameters)
    {
        NVSDK_NGX_CUDA_DestroyParameters(m_ngxParameters);
    }
    NVSDK_NGX_CUDA_Shutdown();
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

cuda_api_impl* p_cuda_api_impl = nullptr;

#if !defined(_WIN32)
__attribute__ ((visibility("default")))
#endif
API_BOOL  rtx_video_api_cuda_create(void* cuContext, void* cuStream, int iGpu, API_BOOL THDREnable, API_BOOL VSREnable)
{
    if (!p_cuda_api_impl)
    {
        p_cuda_api_impl = new cuda_api_impl;
    }
    if (!p_cuda_api_impl) return API_BOOL_FAIL;
    return p_cuda_api_impl->create(cuContext, cuStream, iGpu, THDREnable, VSREnable);
}

#if !defined(_WIN32)
__attribute__((visibility("default")))
#endif
API_BOOL rtx_video_api_cuda_evaluate(uint64_t cuTexObject_Input, uint64_t cuSurfObject_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    if (!p_cuda_api_impl) return API_BOOL_FAIL;
    return p_cuda_api_impl->evaluate(cuTexObject_Input, cuSurfObject_Output, inputRect, outputRect, pVSRSetting, pTHDRSetting);
}

#if !defined(_WIN32)
__attribute__((visibility("default")))
#endif
API_BOOL rtx_video_api_cuda_evaluate_deviceptr(void* cuDeviceptr_Input, void* cuDeviceptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    if (!p_cuda_api_impl) return API_BOOL_FAIL;
    return p_cuda_api_impl->evaluate_deviceptr(cuDeviceptr_Input, cuDeviceptr_Output, inputRect, outputRect, pVSRSetting, pTHDRSetting);
}

#if !defined(_WIN32)
__attribute__((visibility("default")))
#endif
API_BOOL rtx_video_api_cuda_evaluate_hostptr(void* hostptr_Input, void* hostptr_Output, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    if (!p_cuda_api_impl) return API_BOOL_FAIL;
    return p_cuda_api_impl->evaluate_hostptr(hostptr_Input, hostptr_Output, inputRect, outputRect, pVSRSetting, pTHDRSetting);
}

#if !defined(_WIN32)
__attribute__((visibility("default")))
#endif
void rtx_video_api_cuda_shutdown()
{
    if (p_cuda_api_impl)
    {
        p_cuda_api_impl->shutdown();
        delete p_cuda_api_impl;
        p_cuda_api_impl = nullptr;
    }
}
