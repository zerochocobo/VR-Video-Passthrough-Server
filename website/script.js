const header = document.querySelector(".site-header");
const revealTargets = document.querySelectorAll(".section-reveal");

const imageByLanguage = {
  intro: {
    en: "assets/intro_en_s.png",
    zh: "assets/intro_cn_s.png",
    ja: "assets/intro_jp_s.png",
  },
  heroScreen: {
    en: "assets/soft_mainwindow_en.png",
    zh: "assets/soft_mainwindow_cn.png",
    ja: "assets/soft_mainwindow_jp.png",
  },
};

const translations = {
  en: {
    "meta.title": "VR Video Passthrough Server",
    "meta.description": "VR Video Passthrough Server brings passthrough viewing to local VR videos.",
    "brand.local": "VR Video Passthrough Server",
    "brand.sub": "",
    "nav.features": "Features",
    "nav.gallery": "Gallery",
    "nav.download": "Download",
    "hero.eyebrow": "A mixed reality video tool for VR180 and local media libraries",
    "hero.title": "Give your VR videos a passthrough viewing experience",
    "hero.text": "Run a local DLNA media server on Windows, stream videos to Quest and other VR players, and generate green-screen or Alpha passthrough output in realtime.",
    "hero.primary": "Download",
    "hero.secondary": "View Results",
    "hero.badge": "Realtime passthrough stream",
    "release.versionLabel": "Current version",
    "release.dateLabel": "Release date",
    "features.eyebrow": "No need to rebuild your video library",
    "features.title": "Turn local VR videos into a passthrough-ready media library",
    "features.text": "Your PC handles the video processing. Your VR player discovers and plays the library over your home network.",
    "featureCards.discovery.title": "Automatic VR player discovery",
    "featureCards.discovery.text": "Expose local video folders through DLNA/UPnP for common VR players such as Skybox, Moon Player, 4XVR, DeoVR, and HereSphere.",
    "featureCards.realtime.title": "Realtime passthrough output",
    "featureCards.realtime.text": "Use GPU matting and HEVC encoding to convert normal VR video into green-screen or Alpha passthrough streams.",
    "featureCards.subtitles.title": "Realtime subtitle embedding",
    "featureCards.subtitles.text": "Keep subtitles visible in passthrough playback, with desktop preview and style controls.",
    "featureCards.offline.title": "Offline video generation",
    "featureCards.offline.text": "Pre-generate passthrough videos when you want saved output or less realtime load while watching.",
    "featureCards.ui.title": "Desktop control",
    "featureCards.ui.text": "Start the server, select folders, switch output modes, and manage settings from a Chinese, English, or Japanese desktop UI.",
    "featureCards.performance.icon": "Light",
    "featureCards.performance.title": "Lighting",
    "featureCards.performance.text": "Adjust video to match ambient light, with three presets and full custom control so content blends more naturally into the real environment.",
    "featureCards.twoDvr.title": "Realtime 2D to 3D",
    "featureCards.twoDvr.text": "Convert flat 2D videos into stereo 3D output for VR playback, with DA3 depth estimation and GPU rendering for live and offline workflows.",
    "featureCards.si.title": "Simultaneous interpretation",
    "featureCards.si.text": "Add same-stem interpretation audio as a .si.wav sidecar and play a mixed [SI] virtual MP4 entry from DLNA.",
    "featureCards.naming.title": "VR player filename compatibility",
    "featureCards.naming.text": "Generated and virtual titles automatically use VR-friendly naming patterns so common players recognize stereo, fisheye, alpha, and live entries correctly.",
    "gallery.eyebrow": "Output examples",
    "gallery.title": "Choose the passthrough format your player supports",
    "gallery.text": "Alpha works best with players that support transparency. Green-screen output works with ChromaKey playback.",
    "requirements.eyebrow": "Requirements",
    "requirements.title": "Designed for local Windows PCs",
    "requirements.windows": "Primary runtime platform",
    "requirements.gpu": "RTX 20 series or newer is recommended. Realtime processing is recommended with 6GB+ VRAM.",
    "requirements.ffmpeg": "Used for media probing, encoding, and muxing.",
    "requirements.quest": "Tested on Quest 3 and Pico 4, and usable with compatible DLNA passthrough players.",
    "compatibility.eyebrow": "Player compatibility",
    "compatibility.title": "Match the output mode to your VR player",
    "compatibility.text": "Support varies by player and playback mode. Choose Alpha when your player supports it, or use ChromaKey green-screen mode as the broader fallback.",
    "compatibility.player": "Player",
    "compatibility.alpha": "Alpha passthrough",
    "compatibility.gray": "Gray green screen",
    "compatibility.chroma": "ChromaKey green screen",
    "download.eyebrow": "Get started",
    "download.title": "Download the package and connect your VR player",
    "download.text": "Run it on your Windows PC, choose your local video folders, then find the server from your VR player through DLNA.",
    "download.githubRelease": "Download from GitHub Releases",
    "download.quark": "Quark Cloud Drive",
    "download.baidu": "Baidu Netdisk",
    "download.source": "GitHub source code",
  },
  zh: {
    "meta.title": "VR视频透视服务器",
    "meta.description": "VR视频透视服务器是一款面向 Windows 的 VR 视频透视服务器，让普通 VR 视频也能进入混合现实观看体验。",
    "brand.local": "VR视频透视服务器",
    "brand.sub": "VR Video Passthrough Server",
    "nav.features": "功能",
    "nav.gallery": "效果",
    "nav.download": "下载",
    "hero.eyebrow": "面向 VR180 与本地媒体库的混合现实视频工具",
    "hero.title": "让你的 VR 视频拥有透视观看体验",
    "hero.text": "在 Windows 电脑上运行本地 DLNA 媒体服务器，把视频发送到 Quest 等 VR 播放器，并实时生成绿幕或 Alpha 透视视频流。",
    "hero.primary": "立即下载",
    "hero.secondary": "查看效果",
    "hero.badge": "实时透视流",
    "release.versionLabel": "当前版本",
    "release.dateLabel": "发布日期",
    "features.eyebrow": "不用重做视频库",
    "features.title": "把本地 VR 视频变成可透视播放的媒体库",
    "features.text": "电脑负责处理视频，VR 播放器通过局域网发现并播放。",
    "featureCards.discovery.title": "VR 播放器自动发现",
    "featureCards.discovery.text": "通过 DLNA/UPnP 暴露本地视频目录，兼容 Skybox、Moon Player、4XVR、DeoVR、HereSphere 等常见播放器。",
    "featureCards.realtime.title": "实时透视输出",
    "featureCards.realtime.text": "使用 GPU 抠像与 HEVC 编码，把普通 VR 视频实时转换为绿幕或 Alpha 透视流。",
    "featureCards.subtitles.title": "字幕实时嵌入",
    "featureCards.subtitles.text": "观看透视视频时也能保留字幕，并在桌面端预览和调整字幕样式。",
    "featureCards.offline.title": "离线生成视频",
    "featureCards.offline.text": "需要提前处理或长期保存时，可以生成离线透视视频文件，减少播放时的实时负担。",
    "featureCards.ui.title": "桌面化控制",
    "featureCards.ui.text": "提供中文、英文、日文界面，启动服务器、选择目录、切换输出模式都在桌面窗口内完成。",
    "featureCards.performance.icon": "光照",
    "featureCards.performance.title": "光照",
    "featureCards.performance.text": "可按照环境光调整视频，有三种预设和完全自定义，使视频内容更加融合到现实环境。",
    "featureCards.twoDvr.title": "实时 2D 转 3D",
    "featureCards.twoDvr.text": "将普通 2D 视频转换为适合 VR 播放的立体 3D 输出，使用 DA3 深度估计和 GPU 渲染，支持实时播放和离线生成。",
    "featureCards.si.title": "同声传译",
    "featureCards.si.text": "把同名传译音频作为 .si.wav sidecar 放在视频旁边，即可在 DLNA 中播放混音后的 [SI] 虚拟 MP4 入口。",
    "featureCards.naming.title": "VR 播放样式文件名自动兼容",
    "featureCards.naming.text": "生成文件和虚拟标题会自动使用 VR 播放器友好的命名规则，让常见播放器正确识别立体、鱼眼、Alpha 和实时播放入口。",
    "gallery.eyebrow": "输出效果",
    "gallery.title": "选择适合播放器的透视格式",
    "gallery.text": "Alpha 适合支持透明通道的播放器，绿幕适合使用 ChromaKey 的播放方式。",
    "requirements.eyebrow": "运行环境",
    "requirements.title": "为本地 Windows 电脑设计",
    "requirements.windows": "主要运行平台",
    "requirements.gpu": "建议 RTX 20 系列或更新，实时处理建议 6GB 以上显存。",
    "requirements.ffmpeg": "用于媒体探测、编码与封装。",
    "requirements.quest": "已在 Quest 3 和 Pico 4 测试，也可配合支持 DLNA 与透视格式的播放器使用。",
    "compatibility.eyebrow": "播放器兼容性",
    "compatibility.title": "根据 VR 播放器选择输出模式",
    "compatibility.text": "不同播放器对模式的支持不同。播放器支持 Alpha 时优先使用 Alpha；需要更通用的方案时可使用 ChromaKey 绿幕模式。",
    "compatibility.player": "播放器",
    "compatibility.alpha": "Alpha 透视",
    "compatibility.gray": "灰色绿幕",
    "compatibility.chroma": "ChromaKey 绿幕",
    "download.eyebrow": "开始使用",
    "download.title": "下载整合包，连接你的 VR 播放器",
    "download.text": "下载后在 Windows 电脑上运行，选择本地视频目录，在 VR 播放器中通过 DLNA 找到服务器即可浏览视频。",
    "download.githubRelease": "GitHub Releases 下载",
    "download.quark": "夸克网盘下载",
    "download.baidu": "百度网盘下载",
    "download.source": "GitHub 源代码地址",
  },
  ja: {
    "meta.title": "VR Video Passthrough Server",
    "meta.description": "VR Video Passthrough Server は、ローカルの VR 動画をパススルー視聴に対応させる Windows 向けツールです。",
    "brand.local": "VR動画パススルーサーバー",
    "brand.sub": "VR Video Passthrough Server",
    "nav.features": "機能",
    "nav.gallery": "効果",
    "nav.download": "ダウンロード",
    "hero.eyebrow": "VR180 とローカルメディアライブラリ向けの複合現実ビデオツール",
    "hero.title": "VR 動画にパススルー視聴体験を",
    "hero.text": "Windows PC でローカル DLNA メディアサーバーを起動し、Quest などの VR プレイヤーへ動画を配信しながら、グリーンスクリーンまたは Alpha パススルーをリアルタイム生成します。",
    "hero.primary": "ダウンロード",
    "hero.secondary": "効果を見る",
    "hero.badge": "リアルタイムパススルー",
    "release.versionLabel": "現在のバージョン",
    "release.dateLabel": "リリース日",
    "features.eyebrow": "動画ライブラリを作り直す必要はありません",
    "features.title": "ローカル VR 動画をパススルー対応メディアライブラリへ",
    "features.text": "PC が動画処理を担当し、VR プレイヤーはホームネットワーク経由でライブラリを検出して再生します。",
    "featureCards.discovery.title": "VR プレイヤーの自動検出",
    "featureCards.discovery.text": "DLNA/UPnP 経由でローカル動画フォルダを公開し、Skybox、Moon Player、4XVR、DeoVR、HereSphere などの一般的な VR プレイヤーで利用できます。",
    "featureCards.realtime.title": "リアルタイムパススルー出力",
    "featureCards.realtime.text": "GPU マッティングと HEVC エンコードにより、通常の VR 動画をグリーンスクリーンまたは Alpha パススルーストリームへ変換します。",
    "featureCards.subtitles.title": "字幕のリアルタイム埋め込み",
    "featureCards.subtitles.text": "パススルー再生中も字幕を表示でき、デスクトップ側でプレビューとスタイル設定ができます。",
    "featureCards.offline.title": "オフライン動画生成",
    "featureCards.offline.text": "保存用の出力が必要な場合や再生時の負荷を下げたい場合に、事前にパススルー動画を生成できます。",
    "featureCards.ui.title": "デスクトップ操作",
    "featureCards.ui.text": "中国語、英語、日本語対応のデスクトップ UI から、サーバー起動、フォルダ選択、出力モード切替、設定管理ができます。",
    "featureCards.performance.icon": "照明",
    "featureCards.performance.title": "照明",
    "featureCards.performance.text": "環境光に合わせて映像を調整できます。3つのプリセットと完全なカスタム設定により、映像コンテンツを現実環境へより自然になじませます。",
    "featureCards.twoDvr.title": "リアルタイム 2D→3D",
    "featureCards.twoDvr.text": "通常の 2D 動画を VR 再生向けのステレオ 3D 出力へ変換します。DA3 深度推定と GPU レンダリングにより、リアルタイム再生とオフライン生成に対応します。",
    "featureCards.si.title": "同時通訳",
    "featureCards.si.text": "同名の通訳音声を .si.wav サイドカーとして置くと、DLNA からミックス済みの [SI] 仮想 MP4 エントリを再生できます。",
    "featureCards.naming.title": "VR 再生用ファイル名の自動互換",
    "featureCards.naming.text": "生成ファイルと仮想タイトルに VR プレイヤー向けの命名規則を自動適用し、ステレオ、魚眼、Alpha、ライブ項目を正しく認識させます。",
    "gallery.eyebrow": "出力例",
    "gallery.title": "プレイヤーに合うパススルー形式を選択",
    "gallery.text": "Alpha は透明チャンネル対応プレイヤーに適し、グリーンスクリーンは ChromaKey 再生に適しています。",
    "requirements.eyebrow": "動作環境",
    "requirements.title": "ローカル Windows PC 向けに設計",
    "requirements.windows": "主な実行プラットフォーム",
    "requirements.gpu": "RTX 20 シリーズ以降を推奨します。リアルタイム処理には 6GB 以上の VRAM を推奨します。",
    "requirements.ffmpeg": "メディア解析、エンコード、mux に使用します。",
    "requirements.quest": "Quest 3 と Pico 4 でテスト済みです。互換性のある DLNA パススループレイヤーでも利用できます。",
    "compatibility.eyebrow": "プレイヤー互換性",
    "compatibility.title": "VR プレイヤーに合う出力モードを選択",
    "compatibility.text": "対応状況はプレイヤーと再生モードによって異なります。Alpha 対応プレイヤーでは Alpha を使い、より広い互換性が必要な場合は ChromaKey グリーンスクリーンを使用します。",
    "compatibility.player": "プレイヤー",
    "compatibility.alpha": "Alpha パススルー",
    "compatibility.gray": "グレーグリーンスクリーン",
    "compatibility.chroma": "ChromaKey グリーンスクリーン",
    "download.eyebrow": "始めましょう",
    "download.title": "パッケージをダウンロードして VR プレイヤーに接続",
    "download.text": "Windows PC で起動し、ローカル動画フォルダを選択してから、VR プレイヤーの DLNA 画面でサーバーを見つけて再生します。",
    "download.githubRelease": "GitHub Releases からダウンロード",
    "download.quark": "Quark Cloud Drive",
    "download.baidu": "Baidu Netdisk",
    "download.source": "GitHub ソースコード",
  },
};

const normalizeLanguage = (language) => {
  const value = String(language || "").toLowerCase();
  if (value.startsWith("zh")) return "zh";
  if (value.startsWith("ja")) return "ja";
  return "en";
};

const getInitialLanguage = () => {
  const requested = new URLSearchParams(window.location.search).get("lang");
  if (requested && translations[normalizeLanguage(requested)]) {
    return normalizeLanguage(requested);
  }
  const saved = localStorage.getItem("ptserver-language");
  if (saved && translations[saved]) return saved;
  return normalizeLanguage(navigator.language || navigator.userLanguage);
};

const translate = (language, key) => translations[language][key] || translations.en[key] || "";

const applyLanguage = (language) => {
  const lang = translations[language] ? language : "en";
  document.documentElement.lang = lang === "zh" ? "zh-CN" : lang;
  document.title = translate(lang, "meta.title");

  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = translate(lang, element.dataset.i18n);
  });

  document.querySelectorAll("[data-i18n-attr]").forEach((element) => {
    element.dataset.i18nAttr.split(",").forEach((entry) => {
      const [attr, key] = entry.split(":").map((part) => part.trim());
      element.setAttribute(attr, translate(lang, key));
    });
  });

  Object.entries(imageByLanguage).forEach(([name, sources]) => {
    const image = document.querySelector(`[data-lang-image="${name}"]`);
    if (image) image.src = sources[lang] || sources.en;
  });

  document.querySelectorAll("[data-lang-button]").forEach((button) => {
    button.setAttribute("aria-pressed", button.dataset.langButton === lang ? "true" : "false");
  });

  localStorage.setItem("ptserver-language", lang);
};

const updateHeader = () => {
  header.dataset.scrolled = window.scrollY > 12 ? "true" : "false";
};

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.16 }
);

document.querySelectorAll("[data-lang-button]").forEach((button) => {
  button.addEventListener("click", () => applyLanguage(button.dataset.langButton));
});

revealTargets.forEach((target) => observer.observe(target));
window.addEventListener("scroll", updateHeader, { passive: true });
applyLanguage(getInitialLanguage());
updateHeader();
