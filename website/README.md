# Thru3D 官网

`website/` 是可直接部署的静态站点，无构建步骤、无第三方字体或前端依赖。入口 `index.html`；已发布隐私政策始终保留 `privacy.html`。

## 本地预览

在仓库根目录运行：

```powershell
python -m http.server 8785 --bind 127.0.0.1 --directory website
```

打开 http://127.0.0.1:8785/ 。中英日可通过语言按钮切换，或使用 `?lang=en`、`?lang=zh`、`?lang=ja`；语言偏好与隐私页共享。

## 产品与发布入口

- Player 与 Server 是独立产品，可通过局域网 DLNA 配合；Player 的服务器透视流使用 Alpha。
- Player Meta 商店：https://www.meta.com/experiences/thru3d-media-player/2508393509572009 。当前状态等待上线，不提供虚构 APK 链接。
- Player 源码：https://github.com/zerochocobo/Thru3D-Media-Player 。QUEST、PICO、实验性 OpenXR 分别展示，不将单设备测试当成全平台验收。
- Server 源码：https://github.com/zerochocobo/Thru3D-Media-Server 。下载链接指向新仓库 `/releases/latest`，保留原夸克与百度镜像。
- 上线后同步更新 `index.html` 的默认英文展示及 `script.js` 三语 `download.early` / `download.playerNote` 文案。链接在 HTML 内维护。
- 版本与日期以各产品公开 Releases 为准，首页没有将开发工作区版本硬编码为已发布版本。
- 此轮只完成本地网站源码；没有部署到 wapok.com 或推送 GitHub。

## 素材与维护

- `assets/thru3d-icon.png` 源自用户指定 Player 项目 `docs/store/meta/assets/icon-512.png`，缩小至 192×192。
- `assets/player-photo.jpg`、`player-library.jpg`、`player-video.jpg` 源自同目录下的 `screenshots/01-panorama-photo.png`、`03-media-server.png`、`05-3d-concert-passthrough-control.png`，缩小至 1600×900 并转为 JPEG。保留真实画面，不把演唱会控制截图声明为最终抠像结果。
- Server 使用已有 `soft_mainwindow_{en,cn,jp}.png`；随网站语言切换。
- 页面的 CSS 光轨与网格为本地代码绘制；没有新增网络图片、字体、统计服务或表单。
- 隐私政策只调整图标、品牌导航和 CSS，政策正文与生效日期保持。

## 验证

`node --check website/script.js`、`node --check website/privacy.js`、`git diff --check`。

浏览器检查证据保存在 Git 忽略的 `debug_output/website_20261009/`：桌面/移动三语截图、交互检查脚本及 `verification.json`。验证包含 1440/1024/768/390/320 宽度、资源加载、三语与隐私导航、截图键盘切换/放大、移动菜单、兼容表及 FAQ。浏览器使用本机 Edge，不要求安装 Playwright 浏览器。