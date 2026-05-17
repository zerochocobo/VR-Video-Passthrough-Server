# VR Video Passthrough Server

日本語 | [English](README.md) | [中文](README.zh-CN.md)

VR Video Passthrough Server の目標は、すべてのVR動画をパススルー対応にし、複合現実（MR）を実現することです。

![VR Video Passthrough Server 概要](assets/intro_jp_s.png)

これは Windows を主な実行環境とする VR DLNA ローカルメディアサーバーで、デスクトップ操作とオフライン生成ワークフローに対応しています。DLNA/UPnP 経由でローカル動画ライブラリを公開し、リアルタイムのパススルーストリーム出力をサポートします。グリーンスクリーン合成と Alpha パススルーを切り替えられ、リアルタイム字幕埋め込みにも対応しています。VR Video Passthrough Server は主に VR180 half-equirectangular 動画ソース向けに設計されています。

## プロジェクトの起源

当初ただVR動画の透視ツールを作りたかった。  
誰かに「車輪の再発明だ」と言われたので、こう返しました。  
「あの何年も前の古い車輪はもう古すぎる。新しいものが必要だ」と。  
7日後、新しい車輪が誕生した。  
これこそ、AI時代の奇跡である。   

## 機能

- DLNA 検出と ContentDirectory ブラウズ
- GPU マッティングと HEVC 出力によるリアルタイムパススルーストリーミング
- パススルーストリームへのリアルタイム字幕埋め込み
- グリーンスクリーンモードと Alpha パススルーモード
- オフラインパススルー動画生成
- 複数のローカル動画ルートディレクトリに対応
- 中国語、英語、日本語に対応した PySide6 デスクトップ UI
- 字幕プレビューと字幕スタイル設定
- ハードウェアが維持できる範囲で 8K クラスのソース再生を目指す、VRAM を意識した積極的なパイプライン調整


| ![MainWindow](assets/soft_mainwindow_jp.png) |


## パススルー出力例

| Alpha Passthrough | グリーンスクリーン Passthrough |
| --- | --- |
| ![Alpha Passthrough の例](assets/sample_alpha.jpg) | ![グリーンスクリーン Passthrough の例](assets/sample_green.jpg) |
| ![Screenshot](assets/passthrough_screenshot.jpg) |

## 動作要件

- Windows 10 / 11
- Python 3.12
- リアルタイムパイプライン用の NVIDIA GPU。目安として RTX 20 シリーズ以上を推奨します。正確な型番は NVIDIA 公式リストを確認してください: <https://developer.nvidia.com/cuda/gpus>。推奨 VRAM: リアルタイムサーバーと RVM オフライン生成は 6 GB 以上、MatAnyone2 / SAM3 オフラインワークフローは約 15 GB 以上。
- FFmpeg / FFprobe

## クイックスタート

```bash
uv run python main.py
```

デスクトップ UI を起動:

```bash
uv run python -m ui.app
```

## 対応 VR 動画プレイヤー

Meta Quest 3 でテストしています。

| プレイヤー | Alpha パススルー | グレーグリーンスクリーン | ChromaKey グリーンスクリーン | Web サイト | 備考 |
| --- | --- | --- | --- | --- | --- |
| Skybox VR Player 2.0.2 Preview | 対応 | - | 対応 | [公式サイト](https://skybox.xyz) | [インストール説明](https://forum.skybox.xyz/d/2920-skybox-quest-v202-preview-performance-improvements) |
| Moon Player | - | 対応 | 対応 | [公式サイト](https://moonvrplayer.com) | - |
| 4XVR Video Player | 対応 | - | 対応 | [公式サイト](https://www.4xvr.net/) | - |
| DeoVR player | 対応 | - | 対応 | [公式サイト](https://deovr.com/) | - |
| HereSphere VR Video Player | 対応 | - | 対応 | [公式サイト](https://heresphere.com/) | - |

## 設定メモ

- `PT_VIDEO_DIR` は `|` 区切りで複数のルートディレクトリを指定できます
- `PT_PASSTHROUGH_OUTPUT_MODE` は `none`、`green`、`alpha`、`all` に対応しています
- Alpha モードでは DLNA 仮想アイテムのタイトルとして `Alpha Passthrough` が使われます
- UI 設定はバックエンドのランタイム設定とは別に保存されます

## プロジェクト構成

```text
main.py        サーバーのエントリポイント
config.py      ランタイム設定
dlna/          UPnP / DLNA 検出とカタログ
http_app/      FastAPI ルート
pipeline/      デコード、マッティング、エンコード、サムネイル、字幕パイプライン
offline/       本番向けオフライン変換エントリポイント
ui/            PySide6 デスクトップ UI、ページ、i18n、プロセス制御
tools/         開発用プローブと診断ツール
models/        ローカルモデルファイルとマニフェスト
resources/     パッケージ用 UI / ランタイムアセット
prompt/        引き継ぎメモと調査レポート
```

## 参照しているオープンソースモデル

VR Video Passthrough Server 自体はマッティングモデルを学習しません。以下の上流プロジェクトが提供するモデルとモデルファイルを使用します。

| モデル | 役割 | 上流 |
| --- | --- | --- |
| Robust Video Matting (RVM) | `rvm_mobilenetv3_fp32.onnx` と `rvm_resnet50_fp32.onnx` を含む、主要なリアルタイムマッティング経路 | [GitHub](https://github.com/PeterL1n/RobustVideoMatting) |
| MatAnyone2 | オフライン変換と実験的ワークフローで使う、低速だが通常は高品質なマッティング経路 | [GitHub](https://github.com/pq-yang/MatAnyone2) |
| Segment Anything Model 3 (SAM 3) | 実験的な Alpha ツールと前処理ワークフローで使う任意の補助モデル | [GitHub](https://github.com/facebookresearch/sam3) |

## 参照している依存関係

- [PySide6](https://www.qt.io/qt-for-python)
- [FastAPI](https://github.com/fastapi/fastapi)
- [Uvicorn](https://github.com/encode/uvicorn)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [CuPy](https://github.com/cupy/cupy)
- [PyNvVideoCodec](https://github.com/NVIDIA/VideoProcessingFramework)

## 注意事項

- このコードベースは、ホスト型サービスとしてのデプロイではなく、ローカル Windows マシンでの利用を主に想定しています。
- Alpha パススルーは `VR Passthrough Server` という DLNA 仮想アイテムとして表示されます。
- 現在のパイプラインは、汎用的な 360 度動画や平面動画ではなく、VR180 half-equirectangular ソース向けに調整されています。
- 英語版は [README.md](README.md)、中国語版は [README.zh-CN.md](README.zh-CN.md) を参照してください。

## ライセンス

ライセンス: `AGPL-3.0-or-later`

プロジェクトのライセンス条件は、リポジトリのライセンスファイルを参照してください。上流モデルのリポジトリには、それぞれ独自のライセンスと利用条件があります。
