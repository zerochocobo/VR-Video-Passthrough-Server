(() => {
  "use strict";

  const translations = {
  "en": {
    "meta.title": "Thru3D — Your media, beyond the screen",
    "meta.description": "Meet Thru3D: an independent VR media player and Windows media server. Enjoy AI passthrough, 2D-to-3D videos and photos, and your own media library.",
    "skip": "Skip to content",
    "brand.home": "Thru3D home",
    "nav.label": "Main navigation",
    "nav.player": "Media Player",
    "nav.server": "Media Server",
    "nav.together": "Better together",
    "nav.download": "Get Thru3D",
    "language": "Language",
    "nav.toggle": "Toggle navigation",
    "hero.eyebrow": "A new dimension for your media",
    "hero.title": "Your media.",
    "hero.accent": "Beyond the screen.",
    "hero.text": "Bring videos into your space. Rediscover photos in 3D. Meet the player and server that open up your own media.",
    "hero.player": "Explore the Player",
    "hero.server": "Meet the Server",
    "hero.independent": "Two independent apps. Even more possibilities together.",
    "scene.label": "/ SPATIAL MEDIA",
    "hero.strip1": "VR & mixed reality",
    "hero.strip2": "AI passthrough & 3D depth",
    "hero.strip3": "Your own media library",
    "alt.photo": "Panoramic photo viewed inside Thru3D Media Player",
    "alt.server": "Thru3D Media Server Windows interface",
    "alt.library": "Media server library in Thru3D Media Player",
    "alt.video": "VR concert video with Thru3D Media Player controls",
    "player.title": "A whole new dimension.",
    "player.accent": "No PC required.",
    "player.intro": "Your standalone VR player for videos and photos. AI runs right on your headset, so you can start with the media you already have.",
    "gallery.label": "Inside Thru3D",
    "gallery.capture": "In-app capture",
    "gallery.enlarge": "Enlarge screenshot",
    "gallery.tabs": "Player screenshots",
    "gallery.library": "Your library",
    "gallery.photo": "Immersive photos",
    "gallery.video": "VR video",
    "gallery.dialog": "Player screenshot",
    "gallery.close": "Close screenshot",
    "player.passthrough.title": "People beyond the screen.",
    "player.passthrough.text": "AI removes the background from supported 180° videos, bringing people into your real surroundings. No green screen needed.",
    "player.depth.title": "Flat media. New depth.",
    "player.depth.text": "Turn everyday 2D videos and photos into stereoscopic 3D. Adjust the depth to find your perfect view.",
    "player.library.title": "Your library, within reach.",
    "player.library.text": "Local files, SMB, WebDAV, DLNA, Emby, Jellyfin, XBVR and Stash. Browse and sign in inside VR.",
    "player.extra1": "Pinch, browse & zoom photos",
    "player.extra2": "Bookmarks & resume playback",
    "player.extra3": "Audio tracks & VR subtitles",
    "player.cta": "Get the Player",
    "server.title": "More possibilities.",
    "server.accent": "Powered by your PC.",
    "server.intro": "Transform your local video library with the power of your GPU. Stream over DLNA to Thru3D or another compatible VR player, or save the results for later.",
    "server.cta": "Get the Server",
    "server.caption": "Local library. Powerful processing.",
    "server.f1.title": "AI passthrough",
    "server.f1.text": "GPU background removal with Alpha or green-screen output.",
    "server.f2.title": "2D → 3D",
    "server.f2.text": "Add stereo depth to flat videos, live or offline.",
    "server.f3.title": "RTX super resolution",
    "server.f3.text": "Enhance supported 2D and VR video with NVIDIA RTX Video.",
    "server.f4.title": "Offline tools",
    "server.f4.text": "Generate videos ahead of time and keep the output.",
    "server.f5.title": "Embedded subtitles",
    "server.f5.text": "Keep subtitles in the stream, with adjustable styles.",
    "server.f6.title": "Interpretation audio",
    "server.f6.text": "Mix your matching interpretation track into playback.",
    "together.eyebrow": "Choose your own setup",
    "together.title": "Great on their own.",
    "together.accent": "Better together.",
    "together.text": "Let your PC handle demanding processing while you enjoy the results in your headset. Your library stays on your devices.",
    "flow.library": "Your video library",
    "flow.librarySub": "On your Windows PC",
    "flow.serverSub": "Process & share",
    "flow.playerSub": "Browse & enjoy",
    "step1.title": "Open your library",
    "step1.text": "Add video folders to Server on your PC.",
    "step2.title": "Choose your output",
    "step2.text": "Use Alpha passthrough for Thru3D Player, or choose another supported mode.",
    "step3.title": "Connect in VR",
    "step3.text": "On the same local network, open Player's DLNA library and select Server.",
    "compat.title": "Using Server with another VR player?",
    "compat.text": "Server also works with compatible DLNA players. Match the output to the player and version you use. Thru3D Player uses Alpha passthrough.",
    "compat.caption": "Passthrough output compatibility",
    "compat.player": "Player",
    "compat.gray": "Gray green screen",
    "supported": "Supported",
    "download.eyebrow": "Your next dimension starts here",
    "download.title": "Find your Thru3D.",
    "download.text": "Pick the app that fits your setup. You can add the other whenever you like.",
    "download.early": "Meta · Coming soon",
    "download.playerText": "Your videos and photos, right in your headset.",
    "experimental": "Experimental",
    "download.playerLink": "View on Meta",
    "download.playerNote": "The Meta store release is pending. Source code and build instructions are available on GitHub.",
    "download.build": "Build guide",
    "download.playerSource": "Player source",
    "download.serverText": "Transform and share your PC video library.",
    "download.serverLink": "Download Server",
    "download.serverNote": "Check the release page for the available version and package details.",
    "download.quark": "Quark Drive",
    "download.baidu": "Baidu Netdisk",
    "download.source": "Source ↗",
    "faq.eyebrow": "A little clarity",
    "faq.title": "Before you begin.",
    "faq.contact": "Get in touch",
    "faq.q1": "Do I need both apps?",
    "faq.a1": "No. Media Player plays local and network media independently, with AI processing on the headset. Media Server works on Windows with compatible DLNA players and offline tools. You can use either app, or combine them.",
    "faq.q2": "Which headsets can I use?",
    "faq.a2": "Player has separate QUEST and PICO builds, targeting Quest 2 and PICO Neo3 or newer. Features and AI performance vary by device and content. The standard OpenXR Android build is experimental and requires a compatible runtime; it is not a promise of support for every headset.",
    "faq.q3": "What does Server need?",
    "faq.a3": "Windows 10 or 11 and an NVIDIA GPU for GPU processing. RTX 20 series or newer and at least 6 GB of VRAM are recommended for common realtime workflows. High-quality offline processing can need substantially more VRAM; see the project documentation for each mode.",
    "faq.q4": "Which media works with the AI features?",
    "faq.a4": "Player's background removal is for supported 180° videos; its depth conversion is for flat mono videos and photos. AI results depend on the source and device. When streaming passthrough from Server to Thru3D Player, choose Alpha output. Server can also generate green-screen output for other compatible players.",
    "footer.tagline": "Your media. Your space. Your way.",
    "footer.privacy": "Privacy Policy",
    "footer.made": "Made for a new dimension."
  },
  "zh": {
    "meta.title": "Thru3D — 让影像走出屏幕",
    "meta.description": "Thru3D Media Player 与 Media Server：两款独立使用、也可搭配的媒体软件。体验 AI 透视、视频与照片 2D 转 3D，以及自己的本地媒体库。",
    "skip": "跳到正文",
    "brand.home": "Thru3D 首页",
    "nav.label": "主导航",
    "nav.player": "媒体播放器",
    "nav.server": "媒体服务器",
    "nav.together": "搭配使用",
    "nav.download": "获取 Thru3D",
    "language": "语言",
    "nav.toggle": "展开或收起导航",
    "hero.eyebrow": "为你的影像，打开新的维度",
    "hero.title": "你的影像，",
    "hero.accent": "不止于屏幕。",
    "hero.text": "让视频走进真实空间，让照片拥有立体深度。用 Thru3D 播放器与服务器，重新发现自己的媒体世界。",
    "hero.player": "探索播放器",
    "hero.server": "了解服务器",
    "hero.independent": "两款独立软件，搭配使用有更多可能。",
    "scene.label": "/ 空间媒体",
    "hero.strip1": "VR 与混合现实",
    "hero.strip2": "AI 透视与立体深度",
    "hero.strip3": "你自己的媒体库",
    "alt.photo": "Thru3D Media Player 内的全景照片观看画面",
    "alt.server": "Thru3D媒体服务器的 Windows 界面",
    "alt.library": "Thru3D Media Player 的媒体服务器资料库界面",
    "alt.video": "Thru3D Media Player 中的 VR 演唱会视频与播放控制",
    "player.title": "观看，进入新维度。",
    "player.accent": "无需连接电脑。",
    "player.intro": "在头显里独立运行的视频与照片播放器。AI 直接在设备上处理，从你已有的影像开始。",
    "gallery.label": "走进 Thru3D",
    "gallery.capture": "应用内截图",
    "gallery.enlarge": "放大截图",
    "gallery.tabs": "播放器截图",
    "gallery.library": "媒体库",
    "gallery.photo": "沉浸式照片",
    "gallery.video": "VR 视频",
    "gallery.dialog": "播放器截图",
    "gallery.close": "关闭截图",
    "player.passthrough.title": "人物，走出屏幕。",
    "player.passthrough.text": "AI 为支持的 180° 视频移除背景，让人物融入眼前的真实环境，无需提前准备绿幕。",
    "player.depth.title": "平面影像，立体新生。",
    "player.depth.text": "让日常 2D 视频和照片呈现立体 3D 深度，随心调整，找到适合自己的观看效果。",
    "player.library.title": "媒体库，触手可及。",
    "player.library.text": "本地文件、SMB、WebDAV、DLNA，以及 Emby、Jellyfin、XBVR、Stash，在 VR 内浏览与登录。",
    "player.extra1": "捏合浏览与双手缩放照片",
    "player.extra2": "时间书签与断点续播",
    "player.extra3": "多音轨与 VR 字幕",
    "player.cta": "获取播放器",
    "server.title": "更多可能，",
    "server.accent": "由电脑驱动。",
    "server.intro": "用 GPU 的算力拓展本地视频库。通过 DLNA 推流到 Thru3D 或其他兼容的 VR 播放器，也能生成文件留待之后观看。",
    "server.cta": "获取服务器",
    "server.caption": "本地媒体库，强大的处理能力。",
    "server.f1.title": "AI 透视",
    "server.f1.text": "GPU 移除背景，输出 Alpha 或绿幕透视。",
    "server.f2.title": "2D → 3D",
    "server.f2.text": "为平面视频增加立体深度，支持实时和离线处理。",
    "server.f3.title": "RTX 超分辨率",
    "server.f3.text": "使用 NVIDIA RTX Video 增强支持的 2D 与 VR 视频。",
    "server.f4.title": "离线工具",
    "server.f4.text": "提前生成处理后的视频，保存输出文件。",
    "server.f5.title": "字幕嵌入",
    "server.f5.text": "将字幕保留在视频流中，自由调整显示样式。",
    "server.f6.title": "传译音轨",
    "server.f6.text": "将配套的传译音频混入视频播放。",
    "together.eyebrow": "自由选择你的观看方式",
    "together.title": "各自出色，",
    "together.accent": "搭配更精彩。",
    "together.text": "让电脑承担繁重的视频处理，在头显中享受结果。媒体库始终保存在你自己的设备上。",
    "flow.library": "你的视频库",
    "flow.librarySub": "保存在 Windows 电脑",
    "flow.serverSub": "处理与分享",
    "flow.playerSub": "浏览与观看",
    "step1.title": "打开媒体库",
    "step1.text": "在电脑的 Server 中添加视频目录。",
    "step2.title": "选择输出模式",
    "step2.text": "搭配 Thru3D Player 时选择 Alpha 透视，也可使用其他支持的模式。",
    "step3.title": "在 VR 中连接",
    "step3.text": "连接同一局域网，在 Player 的 DLNA 媒体库中找到 Server。",
    "compat.title": "想搭配其他 VR 播放器？",
    "compat.text": "Server 也能配合其他兼容 DLNA 的播放器。请按播放器与版本选择输出模式；Thru3D Player 的透视播放使用 Alpha。",
    "compat.caption": "透视输出兼容性",
    "compat.player": "播放器",
    "compat.gray": "灰色绿幕",
    "supported": "支持",
    "download.eyebrow": "从这里，打开新的维度",
    "download.title": "选择你的 Thru3D。",
    "download.text": "先选择适合自己的软件，随时加入另一款。",
    "download.early": "Meta · 等待上线",
    "download.playerText": "在头显里，享受自己的视频与照片。",
    "experimental": "实验性",
    "download.playerLink": "前往 Meta 商店",
    "download.playerNote": "Meta 商店版本正在等待上线。GitHub 提供源码与构建说明。",
    "download.build": "构建指南",
    "download.playerSource": "播放器源码",
    "download.serverText": "处理并分享电脑上的本地视频库。",
    "download.serverLink": "下载服务器",
    "download.serverNote": "可下载版本与整合包信息，以 Releases 页面为准。",
    "download.quark": "夸克网盘",
    "download.baidu": "百度网盘",
    "download.source": "源码 ↗",
    "faq.eyebrow": "开始前，了解更多",
    "faq.title": "你可能想知道。",
    "faq.contact": "联系我们",
    "faq.q1": "必须同时安装两款软件吗？",
    "faq.a1": "不需要。Media Player 能独立播放本地与网络媒体，AI 在头显上运行。Media Server 在 Windows 上处理视频，支持兼容的 DLNA 播放器与离线工具。你可以单独使用任意一款，也可以搭配使用。",
    "faq.q2": "Player 支持哪些头显？",
    "faq.a2": "Player 分别提供 QUEST 与 PICO 构建，最低兼容目标为 Quest 2 与 PICO Neo3。功能与 AI 性能随设备、内容而变化。标准 OpenXR Android 构建为实验性，需要兼容的运行时，不代表所有头显都已验证可用。",
    "faq.q3": "Server 需要什么电脑配置？",
    "faq.a3": "Windows 10 或 11，GPU 处理需要 NVIDIA 显卡。常规实时流程建议 RTX 20 系列或更新、至少 6 GB 显存。高质量离线处理可能需要更多显存，各模式的具体要求请查看项目文档。",
    "faq.q4": "AI 功能适用于哪些影像？",
    "faq.a4": "Player 的背景移除面向支持的 180° 视频，深度转换面向平面单眼视频与照片。AI 效果取决于素材和设备。Server 向 Thru3D Player 推送透视流时请选择 Alpha；Server 也能为其他兼容播放器生成绿幕输出。",
    "footer.tagline": "你的影像，你的空间，你的方式。",
    "footer.privacy": "隐私政策",
    "footer.made": "为新的维度而生。"
  },
  "ja": {
    "meta.title": "Thru3D — メディアを、スクリーンのその先へ",
    "meta.description": "Thru3D Media Player と Windows 向け Media Server。単独でも、一緒でも。AI パススルー、動画・写真の 2D→3D と、自分だけのメディアライブラリを楽しめます。",
    "skip": "本文へ移動",
    "brand.home": "Thru3D ホーム",
    "nav.label": "メインナビゲーション",
    "nav.player": "Media Player",
    "nav.server": "Media Server",
    "nav.together": "組み合わせる",
    "nav.download": "Thru3D を入手",
    "language": "言語",
    "nav.toggle": "ナビゲーションを開閉",
    "hero.eyebrow": "あなたのメディアに、新しい次元を",
    "hero.title": "メディアを、",
    "hero.accent": "その先の空間へ。",
    "hero.text": "動画を自分の空間に。写真に新しい奥行きを。Thru3D のプレイヤーとサーバーで、いつものメディアを再発見。",
    "hero.player": "Player を見る",
    "hero.server": "Server を見る",
    "hero.independent": "2 つの独立したアプリ。一緒に使えば、可能性が広がります。",
    "scene.label": "/ 空間メディア",
    "hero.strip1": "VR と複合現実",
    "hero.strip2": "AI パススルーと 3D 深度",
    "hero.strip3": "自分だけのライブラリ",
    "alt.photo": "Thru3D Media Player で表示したパノラマ写真",
    "alt.server": "Thru3D Media Server の Windows 画面",
    "alt.library": "Thru3D Media Player のメディアサーバーライブラリ",
    "alt.video": "Thru3D Media Player の VR コンサート動画と再生コントロール",
    "player.title": "新しい次元の視聴へ。",
    "player.accent": "PC は不要。",
    "player.intro": "動画と写真を楽しむ独立型 VR プレイヤー。AI はヘッドセット上で動くので、手持ちのメディアからすぐに始められます。",
    "gallery.label": "Thru3D の中へ",
    "gallery.capture": "アプリ内のスクリーンショット",
    "gallery.enlarge": "画像を拡大",
    "gallery.tabs": "Player の画面",
    "gallery.library": "ライブラリ",
    "gallery.photo": "没入型フォト",
    "gallery.video": "VR 動画",
    "gallery.dialog": "Player のスクリーンショット",
    "gallery.close": "画像を閉じる",
    "player.passthrough.title": "人物を、画面の外へ。",
    "player.passthrough.text": "対応する 180° 動画の背景を AI で除去。人物を現実の空間に重ねて楽しめます。グリーンスクリーンは不要です。",
    "player.depth.title": "平面メディアに、奥行きを。",
    "player.depth.text": "いつもの 2D 動画や写真をステレオ 3D に。深度を調整して、自分に合う見え方に。",
    "player.library.title": "ライブラリを、すぐそばに。",
    "player.library.text": "ローカル、SMB、WebDAV、DLNA、Emby、Jellyfin、XBVR、Stash。VR の中で閲覧・ログインできます。",
    "player.extra1": "ピンチ操作と両手で写真をズーム",
    "player.extra2": "ブックマークと続きから再生",
    "player.extra3": "音声トラックと VR 字幕",
    "player.cta": "Player を入手",
    "server.title": "もっと広がる可能性。",
    "server.accent": "PC の力で。",
    "server.intro": "GPU の力でローカル動画ライブラリを拡張。DLNA で Thru3D や互換 VR プレイヤーへ配信し、処理した動画を保存することもできます。",
    "server.cta": "Server を入手",
    "server.caption": "ローカルライブラリに、強力な処理を。",
    "server.f1.title": "AI パススルー",
    "server.f1.text": "GPU で背景を除去し、Alpha またはグリーンスクリーンで出力。",
    "server.f2.title": "2D → 3D",
    "server.f2.text": "平面動画にステレオ深度を。リアルタイム・オフラインに対応。",
    "server.f3.title": "RTX 超解像",
    "server.f3.text": "NVIDIA RTX Video で対応する 2D・VR 動画を高画質化。",
    "server.f4.title": "オフラインツール",
    "server.f4.text": "事前に動画を処理して、出力ファイルを保存。",
    "server.f5.title": "字幕の埋め込み",
    "server.f5.text": "動画ストリームに字幕を表示。スタイルも調整できます。",
    "server.f6.title": "通訳音声",
    "server.f6.text": "対応する通訳トラックを再生音声にミックス。",
    "together.eyebrow": "自分に合う構成を",
    "together.title": "それぞれでも、",
    "together.accent": "一緒なら、もっと。",
    "together.text": "負荷の高い処理は PC に任せ、結果をヘッドセットで楽しむ。ライブラリは自分のデバイスに保存されます。",
    "flow.library": "動画ライブラリ",
    "flow.librarySub": "Windows PC に保存",
    "flow.serverSub": "処理と共有",
    "flow.playerSub": "閲覧と視聴",
    "step1.title": "ライブラリを開く",
    "step1.text": "PC の Server に動画フォルダーを追加。",
    "step2.title": "出力を選ぶ",
    "step2.text": "Thru3D Player には Alpha パススルーを選択。ほかの対応モードも使えます。",
    "step3.title": "VR で接続する",
    "step3.text": "同じ LAN で Player の DLNA ライブラリを開き、Server を選びます。",
    "compat.title": "ほかの VR プレイヤーと使うには？",
    "compat.text": "Server は互換 DLNA プレイヤーでも使えます。プレイヤーとバージョンに合う出力を選んでください。Thru3D Player のパススルーには Alpha を使用します。",
    "compat.caption": "パススルー出力の互換性",
    "compat.player": "プレイヤー",
    "compat.gray": "グレーグリーンスクリーン",
    "supported": "対応",
    "download.eyebrow": "ここから、新しい次元へ",
    "download.title": "あなたの Thru3D を。",
    "download.text": "自分に合うアプリを選んで、もう一つはいつでも。",
    "download.early": "Meta · 公開準備中",
    "download.playerText": "自分の動画と写真を、ヘッドセットで。",
    "experimental": "実験的",
    "download.playerLink": "Meta ストアを見る",
    "download.playerNote": "Meta ストア版は公開準備中です。ソースコードとビルド手順は GitHub で公開しています。",
    "download.build": "ビルドガイド",
    "download.playerSource": "Player のソース",
    "download.serverText": "PC の動画ライブラリを処理・共有。",
    "download.serverLink": "Server をダウンロード",
    "download.serverNote": "公開バージョンとパッケージの詳細は Releases ページをご確認ください。",
    "download.quark": "Quark Drive",
    "download.baidu": "Baidu Netdisk",
    "download.source": "ソース ↗",
    "faq.eyebrow": "はじめに確認",
    "faq.title": "始める前に。",
    "faq.contact": "お問い合わせ",
    "faq.q1": "両方のアプリが必要ですか？",
    "faq.a1": "いいえ。Media Player はローカル・ネットワークメディアを単独で再生し、AI はヘッドセット上で処理します。Media Server は Windows で動作し、互換 DLNA プレイヤーとオフラインツールを利用できます。単独でも、組み合わせても使えます。",
    "faq.q2": "どのヘッドセットに対応しますか？",
    "faq.a2": "Player は QUEST と PICO に別々のビルドを用意し、最低互換目標は Quest 2 と PICO Neo3 です。機能や AI 性能はデバイス・内容によって変わります。標準 OpenXR Android ビルドは実験的で、互換ランタイムが必要です。すべてのヘッドセットで検証済みという意味ではありません。",
    "faq.q3": "Server に必要な PC は？",
    "faq.a3": "Windows 10 または 11 と、GPU 処理用の NVIDIA GPU が必要です。一般的なリアルタイム処理には RTX 20 シリーズ以降と 6 GB 以上の VRAM を推奨します。高品質なオフライン処理にはより多くの VRAM が必要な場合があります。各モードの要件はプロジェクトの文書をご覧ください。",
    "faq.q4": "AI 機能はどんなメディアで使えますか？",
    "faq.a4": "Player の背景除去は対応する 180° 動画、深度変換は平面モノラル動画と写真が対象です。結果は素材とデバイスによって異なります。Server から Thru3D Player にパススルーを配信するときは Alpha を選択してください。Server はほかの対応プレイヤー向けにグリーンスクリーンも生成できます。",
    "footer.tagline": "あなたのメディア。あなたの空間。あなたらしく。",
    "footer.privacy": "プライバシーポリシー",
    "footer.made": "新しい次元のために。"
  }
};
  const gallery = {
    library: { src: "assets/player-library.jpg", alt: "alt.library", caption: "gallery.library" },
    photo: { src: "assets/player-photo.jpg", alt: "alt.photo", caption: "gallery.photo" },
    video: { src: "assets/player-video.jpg", alt: "alt.video", caption: "gallery.video" },
  };
  const serverImages = {
    en: "assets/soft_mainwindow_en.png",
    zh: "assets/soft_mainwindow_cn.png",
    ja: "assets/soft_mainwindow_jp.png",
  };
  const header = document.querySelector(".site-header");
  const menu = document.querySelector(".top-nav");
  const menuToggle = document.querySelector(".menu-toggle");
  const galleryImage = document.querySelector("[data-gallery-image]");
  const galleryPanel = document.getElementById("experience-panel");
  const galleryTabs = [...document.querySelectorAll("[data-gallery]")];
  const dialog = document.querySelector(".image-dialog");
  let activeLanguage = "en";
  let activeGallery = "library";

  const normalizeLanguage = (value) => {
    const primary = String(value || "").toLowerCase().split(/[-_]/)[0];
    return Object.hasOwn(translations, primary) ? primary : null;
  };
  const translate = (key) => translations[activeLanguage][key] ?? translations.en[key] ?? key;
  const readSavedLanguage = () => {
    try { return normalizeLanguage(localStorage.getItem("ptserver-language")); }
    catch { return null; }
  };
  const initialLanguage = () => normalizeLanguage(new URLSearchParams(location.search).get("lang"))
    || readSavedLanguage() || normalizeLanguage(navigator.language) || "en";

  const setMenu = (open) => {
    menu.dataset.open = String(open);
    menuToggle.setAttribute("aria-expanded", String(open));
  };
  const updateGallery = (name) => {
    if (!Object.hasOwn(gallery, name)) return;
    activeGallery = name;
    const shot = gallery[name];
    galleryImage.src = shot.src;
    galleryImage.alt = translate(shot.alt);
    galleryPanel.setAttribute("aria-labelledby", "tab-" + name);
    galleryTabs.forEach((tab) => {
      const selected = tab.dataset.gallery === name;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
  };
  const updateDialog = () => {
    const shot = gallery[activeGallery];
    const image = dialog.querySelector("[data-dialog-image]");
    image.src = shot.src;
    image.alt = translate(shot.alt);
    dialog.querySelector("[data-dialog-caption]").textContent = translate(shot.caption);
  };
  const applyLanguage = (language, updateUrl = false) => {
    activeLanguage = normalizeLanguage(language) || "en";
    document.documentElement.lang = activeLanguage === "zh" ? "zh-CN" : activeLanguage;
    document.title = translate("meta.title");
    document.querySelectorAll("[data-i18n]").forEach((element) => {
      element.textContent = translate(element.dataset.i18n);
    });
    ["alt", "label", "content"].forEach((kind) => {
      document.querySelectorAll("[data-i18n-" + kind + "]").forEach((element) => {
        const attribute = kind === "label" ? "aria-label" : kind;
        element.setAttribute(attribute, translate(element.getAttribute("data-i18n-" + kind)));
      });
    });
    document.querySelectorAll("[data-server-image]").forEach((image) => {
      image.src = serverImages[activeLanguage];
    });
    document.querySelectorAll("[data-lang-button]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.langButton === activeLanguage));
    });
    document.querySelectorAll("[data-privacy-link]").forEach((link) => {
      link.href = "privacy.html?lang=" + activeLanguage;
    });
    updateGallery(activeGallery);
    if (dialog.open) updateDialog();
    try { localStorage.setItem("ptserver-language", activeLanguage); } catch { /* Optional preference. */ }
    if (updateUrl) {
      const url = new URL(location.href);
      url.searchParams.set("lang", activeLanguage);
      history.replaceState(null, "", url);
    }
  };

  document.querySelectorAll("[data-lang-button]").forEach((button) => {
    button.addEventListener("click", () => applyLanguage(button.dataset.langButton, true));
  });
  menuToggle.addEventListener("click", () => setMenu(menuToggle.getAttribute("aria-expanded") !== "true"));
  menu.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => setMenu(false)));
  document.addEventListener("click", (event) => {
    if (!header.contains(event.target)) setMenu(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && menuToggle.getAttribute("aria-expanded") === "true") {
      setMenu(false);
      menuToggle.focus();
    }
  });
  const desktopMedia = matchMedia("(min-width: 901px)");
  desktopMedia.addEventListener("change", () => setMenu(false));

  galleryTabs.forEach((tab, index) => {
    tab.addEventListener("click", () => updateGallery(tab.dataset.gallery));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % galleryTabs.length;
      if (event.key === "ArrowLeft") next = (index + galleryTabs.length - 1) % galleryTabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = galleryTabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      updateGallery(galleryTabs[next].dataset.gallery);
      galleryTabs[next].focus();
    });
  });
  document.querySelector("[data-open-gallery]").addEventListener("click", () => {
    updateDialog();
    dialog.showModal();
    document.body.classList.add("dialog-open");
  });
  dialog.addEventListener("close", () => document.body.classList.remove("dialog-open"));
  dialog.addEventListener("click", (event) => {
    const bounds = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < bounds.left || event.clientX > bounds.right
      || event.clientY < bounds.top || event.clientY > bounds.bottom)) dialog.close();
  });
  const updateHeader = () => { header.dataset.scrolled = String(scrollY > 12); };
  addEventListener("scroll", updateHeader, { passive: true });
  addEventListener("popstate", () => applyLanguage(initialLanguage()));
  applyLanguage(initialLanguage());
  updateHeader();
})();
