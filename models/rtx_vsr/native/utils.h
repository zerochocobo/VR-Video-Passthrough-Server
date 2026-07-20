//----------------------------------------------------------------------------------
// File:        utils.h
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

#pragma once

#if defined(_WIN32)
#pragma comment( lib, "user32" ) 
#pragma comment( lib, "shell32" )
#pragma comment( lib, "Advapi32" )
#endif

#define APP_ID      0
const wchar_t* pt_rtx_vsr_app_path();
const wchar_t* pt_rtx_vsr_data_path();
#define APP_PATH    pt_rtx_vsr_app_path()

template <class T> inline void SafeRelease(T*& pT)
{
    if (pT != NULL)
    {
        pT->Release();
        pT = NULL;
    }
}

