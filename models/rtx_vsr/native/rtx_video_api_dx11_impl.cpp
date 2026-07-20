//----------------------------------------------------------------------------------
// File:        rtx_video_api_dx11_impl.cpp
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
*  by providing an api taking input and output.
*  Inputs for DX11 are ARGB/ABGR/ABGR10.
*  Output from VSR is in ARGB or if input is 10 bit, ABGR10.
*  Output from THDR is in 10 bit ABGR10.
*  If both are enabled then VSR -> THDR.
*/

#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_defs_truehdr.h>
#include <nvsdk_ngx_helpers_truehdr.h>
#include <nvsdk_ngx_defs_vsr.h>
#include <nvsdk_ngx_helpers_vsr.h>

#if defined(NDEBUG)
#pragma comment( lib, "nvsdk_ngx_s.lib" ) // ngx sdk
#else
#pragma comment( lib, "nvsdk_ngx_s_dbg.lib" ) // ngx sdk
#endif

#include <d3d11_4.h>
#pragma comment( lib, "d3d11" )

#include "rtx_video_api.h"
#include "utils.h"

class dx11_api_impl
{
private:
    ID3D11Device*               m_pD3D11Device          = nullptr;
    ID3D11DeviceContext*        m_pD3D11DeviceContext   = nullptr;
    ID3D10Multithread*          m_pMultiThread          = nullptr;

    bool                        m_bNGXInitialized       = false;
    NVSDK_NGX_Parameter*        m_ngxParameters         = nullptr;
    NVSDK_NGX_Handle*           m_TrueHDRFeature        = nullptr;
    NVSDK_NGX_Handle*           m_VSRFeature            = nullptr;

    bool                        m_bSetupDstTmp          = false;
    ID3D11Texture2D*            m_pDstTmp               = nullptr;
    UINT                        m_uDstTmpWidth          = 0;
    UINT                        m_uDstTmpHeight         = 0;

    bool                        m_bNeedMiddle           = false;
    ID3D11Texture2D*            m_pMiddle               = nullptr;
    UINT                        m_uMiddleWidth          = 0;
    UINT                        m_uMiddleHeight         = 0;


public:
    API_BOOL create(ID3D11Device* pD3DDevice, API_BOOL THDREnable, API_BOOL VSREnable);
    API_BOOL evaluate(ID3D11Texture2D* pInput, ID3D11Texture2D* pOutput, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting);
    void shutdown();
};



API_BOOL dx11_api_impl::create(ID3D11Device* pD3DDevice, API_BOOL THDREnable, API_BOOL VSREnable)
{
    HRESULT hr = S_OK;
    // default to false until creation is done
    m_bNGXInitialized   = false;

    m_pD3D11Device = pD3DDevice;
    m_pD3D11Device->AddRef();

    m_pD3D11Device->GetImmediateContext(&m_pD3D11DeviceContext);

    hr = m_pD3D11DeviceContext->QueryInterface(__uuidof(ID3D10Multithread), (void**)&m_pMultiThread);
    if (SUCCEEDED(hr))
    {
        m_pMultiThread->SetMultithreadProtected(TRUE);
        m_pMultiThread->Enter();
    }

    // init NGX SDK
    NVSDK_NGX_Result NGX_Status = NVSDK_NGX_D3D11_Init(APP_ID, APP_PATH, pD3DDevice);
    if (NVSDK_NGX_FAILED(NGX_Status)) return API_BOOL_FAIL;

    // Get NGX parameters interface (managed and released by NGX)
    NGX_Status = NVSDK_NGX_D3D11_GetCapabilityParameters(&m_ngxParameters);
    if (NVSDK_NGX_FAILED(NGX_Status)) return API_BOOL_FAIL;

    if (THDREnable)
    {
        // Check if TrueHDR is available on the system
        int TrueHDRAvailable = 0;
        NGX_Status = m_ngxParameters->Get(NVSDK_NGX_Parameter_TrueHDR_Available, &TrueHDRAvailable);
        if (!TrueHDRAvailable) return API_BOOL_FAIL;

        // check ScratchBufferSize - truehdr is not expected to request any
        size_t byteSize = 0;
        NGX_Status = NVSDK_NGX_D3D11_GetScratchBufferSize(NVSDK_NGX_Feature_TrueHDR, m_ngxParameters, &byteSize);
        if (NVSDK_NGX_FAILED(NGX_Status)) return API_BOOL_FAIL;
        if (byteSize != 0) return API_BOOL_FAIL;

        // Create the TrueHDR feature instance 
        NVSDK_NGX_Feature_Create_Params TrueHDRCreateParams = {};
        NGX_Status = NGX_D3D11_CREATE_TRUEHDR_EXT(m_pD3D11DeviceContext, &m_TrueHDRFeature, m_ngxParameters, &TrueHDRCreateParams);
        if (NVSDK_NGX_FAILED(NGX_Status)) return API_BOOL_FAIL;
    }
    if (VSREnable)
    {
        // Check if VSR is available on the system
        int VSRAvailable = 0;
        NGX_Status = m_ngxParameters->Get(NVSDK_NGX_Parameter_VSR_Available, &VSRAvailable);
        if (!VSRAvailable) return API_BOOL_FAIL;

        // check ScratchBufferSize - vsr is not expected to request any
        size_t byteSize = 0;
        NGX_Status = NVSDK_NGX_D3D11_GetScratchBufferSize(NVSDK_NGX_Feature_VSR, m_ngxParameters, &byteSize);
        if (NVSDK_NGX_FAILED(NGX_Status)) return API_BOOL_FAIL;
        if (byteSize != 0) return API_BOOL_FAIL;

        // Create the VSR feature instance 
        NVSDK_NGX_Feature_Create_Params VSRCreateParams = {};
        NGX_Status = NGX_D3D11_CREATE_VSR_EXT(m_pD3D11DeviceContext, &m_VSRFeature, m_ngxParameters, &VSRCreateParams);
        if (NVSDK_NGX_FAILED(NGX_Status)) return API_BOOL_FAIL;
    }
    if (m_pMultiThread)
    {
        m_pMultiThread->Leave();
    }

    m_bNeedMiddle = (THDREnable && VSREnable);
    m_bNGXInitialized = true;
    return API_BOOL_SUCCESS;
}

API_BOOL dx11_api_impl::evaluate(ID3D11Texture2D* pInput, ID3D11Texture2D* pOutput, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    if (!m_bNGXInitialized)
    {
        return API_BOOL_FAIL;
    }

    if (m_TrueHDRFeature && !pTHDRSetting)
    {
        return API_BOOL_FAIL;
    }

    if (m_VSRFeature && !pVSRSetting)
    {
        return API_BOOL_FAIL;
    }

    HRESULT hr = S_OK;
    NVSDK_NGX_Result NGX_Status;
    // check formats
    {
        D3D11_TEXTURE2D_DESC inDesc = {};
        D3D11_TEXTURE2D_DESC outDesc = {};
        // check input is DXGI_FORMAT_R8G8B8A8_UNORM or DXGI_FORMAT_B8G8R8A8_UNORM or DXGI_FORMAT_R10G10B10A2_UNORM
        pInput->GetDesc(&inDesc);
        if (inDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM && inDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM && inDesc.Format != DXGI_FORMAT_R10G10B10A2_UNORM)
        {
            return API_BOOL_FAIL;
        }
        // verify input rect is within range
        if (   inputRect.left < 0 || inputRect.left >= inputRect.right  || inputRect.right  > inDesc.Width
            || inputRect.top  < 0 || inputRect.top  >= inputRect.bottom || inputRect.bottom > inDesc.Height)
        {
            return API_BOOL_FAIL;
        }
        pOutput->GetDesc(&outDesc);
        if (m_TrueHDRFeature)
        {
            // check output is HDR format DXGI_FORMAT_R10G10B10A2_UNORM or DXGI_FORMAT_R16G16B16A16_FLOAT
            if (outDesc.Format != DXGI_FORMAT_R10G10B10A2_UNORM && outDesc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT)
            {
                return API_BOOL_FAIL;
            }
        }
        else if (outDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM && outDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM && (inDesc.Format == DXGI_FORMAT_R10G10B10A2_UNORM && outDesc.Format != DXGI_FORMAT_R10G10B10A2_UNORM))
        {
            return API_BOOL_FAIL;
        }

        // verify output rect is within range
        if (   outputRect.left < 0 || outputRect.left >= outputRect.right  || outputRect.right  > outDesc.Width
            || outputRect.top  < 0 || outputRect.top  >= outputRect.bottom || outputRect.bottom > outDesc.Height)
        {
            return API_BOOL_FAIL;
        }

        // The NGX dst surface must be created with BIND_UNORDERED_ACCESS, which swap buffers are not.
        // check for UNORDERED_ACCESS
        m_bSetupDstTmp = !(outDesc.BindFlags & D3D11_BIND_UNORDERED_ACCESS);
 
        // verify DstTmp matches dest surface so copyRegion works
        if (m_bSetupDstTmp && (!m_pDstTmp || outDesc.Width != m_uDstTmpWidth || outDesc.Height != m_uDstTmpHeight))
        {
            SafeRelease(m_pDstTmp);
            m_uDstTmpWidth                          = outDesc.Width;
            m_uDstTmpHeight                         = outDesc.Height;

            D3D11_TEXTURE2D_DESC texture2d_desc     = { 0 };
            texture2d_desc.Width                    = m_uDstTmpWidth;
            texture2d_desc.Height                   = m_uDstTmpHeight;
            texture2d_desc.MipLevels                = 1;
            texture2d_desc.ArraySize                = 1;
            texture2d_desc.SampleDesc.Count         = 1;
            texture2d_desc.MiscFlags                = 0;
            texture2d_desc.Format                   = outDesc.Format;
            texture2d_desc.Usage                    = D3D11_USAGE_DEFAULT;
            texture2d_desc.BindFlags                = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

            hr = m_pD3D11Device->CreateTexture2D(&texture2d_desc, NULL, &m_pDstTmp);
            if (FAILED(hr)) return API_BOOL_FAIL;
        }

        if (m_bNeedMiddle && (!m_pMiddle || outDesc.Width != m_uMiddleWidth || outDesc.Height != m_uMiddleHeight))
        {
            SafeRelease(m_pMiddle);
            m_uMiddleWidth                          = outDesc.Width;
            m_uMiddleHeight                         = outDesc.Height;

            D3D11_TEXTURE2D_DESC texture2d_desc     = { 0 };
            texture2d_desc.Width                    = m_uMiddleWidth;
            texture2d_desc.Height                   = m_uMiddleHeight;
            texture2d_desc.MipLevels                = 1;
            texture2d_desc.ArraySize                = 1;
            texture2d_desc.SampleDesc.Count         = 1;
            texture2d_desc.MiscFlags                = 0;
            texture2d_desc.Format                   = DXGI_FORMAT_R8G8B8A8_UNORM;
            texture2d_desc.Usage                    = D3D11_USAGE_DEFAULT;
            texture2d_desc.BindFlags                = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

            hr = m_pD3D11Device->CreateTexture2D(&texture2d_desc, NULL, &m_pMiddle);
            if (FAILED(hr)) return API_BOOL_FAIL;
        }
    }

    if (m_pMultiThread)
    {
        m_pMultiThread->Enter();
    }
    if (m_VSRFeature)
    {
        NVSDK_NGX_D3D11_VSR_Eval_Params D3D11VsrEvalParams = {};
        D3D11VsrEvalParams.pInput                       = pInput;
        D3D11VsrEvalParams.pOutput                      = m_bNeedMiddle ? m_pMiddle : (m_bSetupDstTmp ? m_pDstTmp : pOutput);
        D3D11VsrEvalParams.InputSubrectBase.X           = inputRect.left;
        D3D11VsrEvalParams.InputSubrectBase.Y           = inputRect.top;
        D3D11VsrEvalParams.InputSubrectSize.Width       = inputRect.right - inputRect.left;
        D3D11VsrEvalParams.InputSubrectSize.Height      = inputRect.bottom - inputRect.top;
        D3D11VsrEvalParams.OutputSubrectBase.X          = outputRect.left;
        D3D11VsrEvalParams.OutputSubrectBase.Y          = outputRect.top;
        D3D11VsrEvalParams.OutputSubrectSize.Width      = outputRect.right - outputRect.left;
        D3D11VsrEvalParams.OutputSubrectSize.Height     = outputRect.bottom - outputRect.top;
        D3D11VsrEvalParams.QualityLevel                 = (NVSDK_NGX_VSR_QualityLevel)pVSRSetting->QualityLevel;

        NGX_Status = NGX_D3D11_EVALUATE_VSR_EXT(m_pD3D11DeviceContext, m_VSRFeature, m_ngxParameters, &D3D11VsrEvalParams);
        if (NVSDK_NGX_FAILED(NGX_Status)) return API_BOOL_FAIL;
    }

    if (m_TrueHDRFeature)
    {
        NVSDK_NGX_D3D11_TRUEHDR_Eval_Params D3D11TrueHDREvalParams = {};

        D3D11TrueHDREvalParams.pInput                   = m_bNeedMiddle ? m_pMiddle : pInput;
        D3D11TrueHDREvalParams.pOutput                  = m_bSetupDstTmp ? m_pDstTmp : pOutput;
        D3D11TrueHDREvalParams.InputSubrectTL.X         = m_bNeedMiddle ? outputRect.left   : inputRect.left;
        D3D11TrueHDREvalParams.InputSubrectTL.Y         = m_bNeedMiddle ? outputRect.top    : inputRect.top;
        D3D11TrueHDREvalParams.InputSubrectBR.Width     = m_bNeedMiddle ? outputRect.right  : inputRect.right;
        D3D11TrueHDREvalParams.InputSubrectBR.Height    = m_bNeedMiddle ? outputRect.bottom : inputRect.bottom;
        D3D11TrueHDREvalParams.OutputSubrectTL.X        = outputRect.left;
        D3D11TrueHDREvalParams.OutputSubrectTL.Y        = outputRect.top;
        D3D11TrueHDREvalParams.OutputSubrectBR.Width    = outputRect.right;
        D3D11TrueHDREvalParams.OutputSubrectBR.Height   = outputRect.bottom;
        D3D11TrueHDREvalParams.Contrast                 = pTHDRSetting->Contrast;
        D3D11TrueHDREvalParams.Saturation               = pTHDRSetting->Saturation;
        D3D11TrueHDREvalParams.MiddleGray               = pTHDRSetting->MiddleGray;
        D3D11TrueHDREvalParams.MaxLuminance             = pTHDRSetting->MaxLuminance;
        NGX_Status = NGX_D3D11_EVALUATE_TRUEHDR_EXT(m_pD3D11DeviceContext, m_TrueHDRFeature, m_ngxParameters, &D3D11TrueHDREvalParams);
        if (NVSDK_NGX_FAILED(NGX_Status)) return API_BOOL_FAIL;
    }


    if (m_bSetupDstTmp)
    {
        m_pD3D11DeviceContext->CopySubresourceRegion(pOutput, 0, 0, 0, 0, m_pDstTmp, 0, NULL);
    }

    if (m_pMultiThread)
    {
        m_pMultiThread->Leave();
    }

    return API_BOOL_SUCCESS;
}

void dx11_api_impl::shutdown()
{
    if (m_bNGXInitialized)
    {
        if (m_VSRFeature)
        {
            NVSDK_NGX_D3D11_ReleaseFeature(m_VSRFeature);
            m_VSRFeature = nullptr;
        }
        if (m_TrueHDRFeature)
        {
            NVSDK_NGX_D3D11_ReleaseFeature(m_TrueHDRFeature);
            m_TrueHDRFeature = nullptr;
        }
        NVSDK_NGX_D3D11_Shutdown1(m_pD3D11Device);
        NVSDK_NGX_D3D11_DestroyParameters(m_ngxParameters);
        m_bNGXInitialized = false;
    }
    SafeRelease(m_pDstTmp);
    SafeRelease(m_pMiddle);
    SafeRelease(m_pMultiThread);
    SafeRelease(m_pD3D11DeviceContext);
    SafeRelease(m_pD3D11Device);
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

dx11_api_impl* p_dx11_api_impl = nullptr;

#if !defined(_WIN32)
__attribute__ ((visibility("default")))
#endif
API_BOOL rtx_video_api_dx11_create(ID3D11Device* pD3DDevice, API_BOOL THDREnable, API_BOOL VSREnable)
{
    if (!p_dx11_api_impl)
    {
        p_dx11_api_impl = new dx11_api_impl;
    }
    if (!p_dx11_api_impl) return false;
    return p_dx11_api_impl->create(pD3DDevice, THDREnable, VSREnable);
}

#if !defined(_WIN32)
__attribute__((visibility("default")))
#endif
API_BOOL rtx_video_api_dx11_evaluate(ID3D11Texture2D* pInput, ID3D11Texture2D* pOutput, API_RECT inputRect, API_RECT outputRect, API_VSR_Setting* pVSRSetting, API_THDR_Setting* pTHDRSetting)
{
    if (!p_dx11_api_impl) return false;
    return p_dx11_api_impl->evaluate(pInput, pOutput, inputRect, outputRect, pVSRSetting, pTHDRSetting);
}

#if !defined(_WIN32)
__attribute__((visibility("default")))
#endif
void rtx_video_api_dx11_shutdown()
{
    if (p_dx11_api_impl)
    {
        p_dx11_api_impl->shutdown();
        delete p_dx11_api_impl;
        p_dx11_api_impl = nullptr;
    }
}
