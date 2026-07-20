#include "rtx_video_api.h"
#include <d3d11.h>
#include <cstdio>
#include <windows.h>
#include <vector>

static const wchar_t* kPath = L"G:\\GIT\\debug\\PTMediaServer\\models\\rtx_vsr\\runtime";
const wchar_t* pt_rtx_vsr_app_path() { return kPath; }
const wchar_t* pt_rtx_vsr_data_path() { return kPath; }

int main()
{
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    D3D_FEATURE_LEVEL level{};
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, &level, &context);
    if (FAILED(hr)) { std::printf("D3D11CreateDevice failed 0x%08lx\n", (unsigned long)hr); return 2; }
    std::printf("device created level=0x%x\n", (unsigned)level);
    API_BOOL ok = rtx_video_api_dx11_create(device, 0, 1);
    std::printf("rtx_video_api_dx11_create=%u\n", ok);
    if (ok)
    {
        D3D11_TEXTURE2D_DESC inDesc{};
        inDesc.Width = 640; inDesc.Height = 360; inDesc.MipLevels = 1; inDesc.ArraySize = 1;
        inDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; inDesc.SampleDesc.Count = 1;
        inDesc.Usage = D3D11_USAGE_DEFAULT; inDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        std::vector<unsigned char> pixels(640 * 360 * 4, 127);
        D3D11_SUBRESOURCE_DATA init{}; init.pSysMem = pixels.data(); init.SysMemPitch = 640 * 4;
        ID3D11Texture2D* input = nullptr;
        hr = device->CreateTexture2D(&inDesc, &init, &input);
        D3D11_TEXTURE2D_DESC outDesc = inDesc; outDesc.Width = 1280; outDesc.Height = 720;
        ID3D11Texture2D* output = nullptr;
        if (SUCCEEDED(hr)) hr = device->CreateTexture2D(&outDesc, nullptr, &output);
        std::printf("textures hr=0x%08lx\n", (unsigned long)hr);
        if (SUCCEEDED(hr))
        {
            API_VSR_Setting setting{0}; API_THDR_Setting thdr{};
            std::printf("evaluate begin\n"); std::fflush(stdout);
            API_RECT inRect{0, 0, 640, 360}; API_RECT outRect{0, 0, 1280, 720};
            API_BOOL eval = rtx_video_api_dx11_evaluate(input, output, inRect, outRect, &setting, &thdr);
            std::printf("evaluate=%u\n", eval);
            context->Flush();
        }
        if (output) output->Release();
        if (input) input->Release();
    }
    rtx_video_api_dx11_shutdown();
    if (context) context->Release();
    if (device) device->Release();
    return ok ? 0 : 3;
}
