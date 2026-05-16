# DLNA Soft-Subtitle Delivery Plan (English)

- Date: 2026-05-16
- Scope: Allow regular MP4 playback via DLNA to carry subtitle tracks with minimal added cost, and as a side effect retire the MKV path that triggers PyNv `SimpleDecoder` stalls.
- Out of scope: Burned-in subtitles, subtitle translation/OCR, UI redesign.

---

## 1. Current State

- `routes_media.py:/media/{name}` uses `FileResponse`/Range and serves **only** the source MP4 bytes. No subtitle channel exists.
- `dlna/content_directory.py:_video_items_from_index()` and `_didl_for()` emit standard DIDL with a single `<res>` (video) plus albumArtURI (thumbnail).
- DLNA itself supports external subtitles through two well-known mechanisms:
  1. **Samsung/SEC extension**: `<sec:CaptionInfo>` / `<sec:CaptionInfoEx>` elements plus HTTP headers `CaptionInfo.sec` / `getCaptionInfo.sec`. Recognised by most smart TVs and some Android DLNA players.
  2. **Standard DIDL approach**: a second `<res protocolInfo="http-get:*:application/x-subrip:*">subURL</res>` inside the same `<item>`.
- Quest3 players in scope: DeoVR / Pigasus / Skybox / SKYBOX. **Their compliance with DLNA subtitle extensions is unverified** — must be the first thing tested.
- Users currently switch to MKV solely to embed soft subtitles, which triggers the `SimpleDecoder` Matroska stall (see `summary/summary_20260516_MKV_PYNV_SIMPLEDECODER_STUCK_ISSUE_CN.md`). If MP4 can carry subtitles, MKV usage drops to zero and the stall disappears.

---

## 2. Strategy

Three-tier rollout, smallest blast radius first:

| Phase | Content | Trigger | Cost |
|-------|---------|---------|------|
| Phase 1 | DLNA subtitle metadata + static `/subs` endpoint | Immediate | Metadata + a few KB of text |
| Phase 2 | On-demand MP4 remux cache (video/audio `-c copy` + `mov_text`) | Phase 1 not recognised by client | First-play 3–10 s, disk ≈ 1× source |
| Phase 3 | MKV → MP4 auto remux (extract embedded subs, share Phase 2 cache) | After Phase 2 lands | Same as Phase 2 |

Each phase ships independently. If Phase N satisfies the target population, do not advance.

---

## 3. Phase 1: DLNA Subtitle Extension (P0, zero codec cost)

### 3.1 Goal
Inform DLNA clients via protocol that an external subtitle exists, without touching the MP4 byte stream.

### 3.2 Subtitle discovery rules
- Same directory, same stem as the video: `name.srt`, `name.ass`, `name.vtt`, `name.ssa`.
- Language tags: `name.zh.srt`, `name.chi.srt`, `name.eng.srt`, `name.zh-CN.ass`.
- Optional subdirectory convention: `name/subs/*.srt` (Jellyfin/Plex compatible).
- When multiple match, sort by language priority (Chinese → English → other). The first becomes the default caption; the rest are attached with `xml:lang`.

### 3.3 New HTTP endpoints
- `GET /subs/{rel}`: serves the subtitle file as-is with Range/HEAD.
  - MIME: srt → `application/x-subrip`, vtt → `text/vtt`, ass → `application/x-ass`.
  - Headers: `Content-Disposition: inline`, `Access-Control-Allow-Origin: *`.
- `HEAD /subs/{rel}`: headers only, lets clients probe size/existence.

### 3.4 ContentDirectory changes
- `dlna/content_directory.py`
  - Add `xmlns:sec="http://www.sec.co.kr/"` to the root DIDL element in `_didl_for()`.
  - `_video_items_from_index()` attaches a `subtitles: list[dict]` field per item: each entry has `url`, `lang`, `type` (srt/ass/vtt), `mime`.
  - `_didl_for()` rendering, when subtitles exist:
    - After the video `<res>`, append:
      ```
      <res protocolInfo="http-get:*:application/x-subrip:*"
           xml:lang="zh-CN">{subURL}</res>
      ```
    - Also append SEC nodes (required by some LG/Samsung players):
      ```
      <sec:CaptionInfoEx sec:type="srt">{subURL}</sec:CaptionInfoEx>
      <sec:CaptionInfo sec:type="srt">{subURL}</sec:CaptionInfo>
      ```

### 3.5 Media route headers
- `routes_media.py:/media/{name}` GET and HEAD:
  - When subtitles are detected for that video, append:
    - `CaptionInfo.sec: <first subtitle URL>`
    - `getCaptionInfo.sec: 1`
  - Range / Content-Length untouched; pure additive headers.

### 3.6 Acceptance
- BubbleUPnP / VLC DMR / Kodi DLNA can enumerate and switch subtitle tracks.
- At least one of DeoVR / Pigasus / Skybox on Quest3 displays the subtitle. The pass/fail of this test gates Phase 2.
- `/media` and `/subs` Range behaviour does not regress.

### 3.7 Risks and fallback
- **Quest player ignores it** → escalate to Phase 2.
- **Header case sensitivity**: `CaptionInfo.sec` is the exact SEC casing — preserve it.
- **Subtitle encoding garble**: sniff utf-8/gbk on disk, transcode to utf-8 on the wire (still Phase 1 scope).

### 3.8 Effort
0.5–1 day.

---

## 4. Phase 2: On-Demand MP4 Remux Cache (P1, compatibility safety net)

### 4.1 Goal
When Phase 1 metadata is ignored by the player, embed the subtitle into the MP4 container as a soft `mov_text` track without re-encoding video/audio.

### 4.2 Cache layout
- Directory: `cache/subbed/` (same level as existing `debug_output/`, `cache/`; reuse base helper if one exists).
- Filename: `<sha1(src.mtime|src.size|sub_paths.mtimes)>.mp4`.
- Index: `cache/subbed/index.json` (path → entry meta), LRU + TTL combined, default cap 50 GB / 30 days.

### 4.3 Trigger and generation
- `routes_media.py:/media/{name}`:
  1. Look up sibling subtitles.
  2. If present and cache miss → invoke ffmpeg synchronously (per-file lock to deduplicate concurrent requests).
  3. On hit → serve cached file with `FileResponse`; the public URL never changes.
- ffmpeg template:
  ```
  ffmpeg -y -nostdin -hide_banner -loglevel error \
    -i "<src.mp4>" \
    -sub_charenc UTF-8 -i "<sub.srt>" \
    -map 0:v -map 0:a? -map 1:s \
    -c copy -c:s mov_text \
    -metadata:s:s:0 language=chi -metadata:s:s:0 title="zh-CN" \
    -movflags +faststart \
    "<cache.mp4>"
  ```
- ASS/SSA: pre-convert `ffmpeg -i sub.ass sub.srt` then embed (mov_text drops styling). When styling matters, only MKV survives — but that defeats the plan, so skip.
- Multi-track: loop `-map 1:s -map 2:s ...` with one `language` metadata per track.

### 4.4 ContentDirectory adjustments
- `_video_items_from_index()` keeps the mp4 item `url` as `/media/{rel}`. The route internally decides source vs cache. Clients see no difference.
- `bitrate` / `size` must reflect the cached file (compute on first miss, then return in DIDL); otherwise Range / progress-bar drifts.

### 4.5 Invalidation and cleanup
- Subtitle mtime/size change → cache invalidates (hash changes, stale file collected by LRU).
- Source mp4 mtime change → same.
- Optional `tools/subbed_cache_gc.py` cleanup script.

### 4.6 Performance expectations
- 8K HEVC 60-second mp4 + srt: remux ~5–10 s (IO-bound).
- 4K H.264 90-minute mp4 + srt: remux ~30–90 s.
- CPU near zero; bottleneck is disk throughput.
- Subsequent Range requests are equivalent to serving the original mp4.

### 4.7 Risks
- **Disk doubles**: GC is mandatory.
- **First-play latency**: may have the user wait on long 8K files. Mitigate with a background pre-remux worker (add it alongside Phase 1; pre-generate during library scan).
- **NPlayer / DeoVR mov_text styling is weak**: acceptable — text subs do not need styling.

### 4.8 Effort
1.5–2 days.

---

## 5. Phase 3: MKV → MP4 Auto Remux (P1, joins the MKV-stall fix)

### 5.1 Goal
Make .mkv files appear in DLNA listings as .mp4 and route them through the Phase 2 cache. Completely bypasses PyNv `SimpleDecoder`'s Matroska stall.

### 5.2 Flow
1. Index scan recognises .mkv, probes embedded subtitle / audio / video tracks.
2. Background remux:
   ```
   ffmpeg -y -i "<src.mkv>" \
     -map 0:v:0 -map 0:a? -map 0:s? \
     -c:v copy -c:a copy -c:s mov_text \
     -movflags +faststart \
     "<cache.mp4>"
   ```
   - Image subtitles (PGS/VOB) cannot become mov_text → drop them, or hand off to a separate OCR worker (default: drop).
   - Non-H.264/HEVC video: `-c:v copy` still works (MP4 supports H.264/HEVC/AV1). If not, fall back to leaving the .mkv as-is (current behaviour with `needs_fix` mark).
3. DLNA listing uses the cached MP4 path with the original stem; outwardly a single mp4.
4. Cache key includes a "src is mkv" tag to avoid colliding with Phase 2 mp4+srt entries.

### 5.3 Relation to the 5/16 MKV-stall plan
- `_hide_passthrough_for_path()` already hides mkv items with bad cues; the SimpleDecoder path no longer ingests bad-cue mkv.
- Phase 3 also moves the well-formed mkv files away, so the PyNv path **never sees Matroska**. This is exactly "Solution F: auto-remux" from the MKV stall summary, merged into the subtitle cache implementation.

### 5.4 Risks
- First library scan may trigger a remux storm → use a serial worker + priority queue.
- Very large MKV libraries strain disk → GC must keep up.

### 5.5 Effort
0.5–1 day on top of Phase 2.

---

## 6. Priority & Milestones

| Priority | Item | Phase | Done when |
|----------|------|-------|-----------|
| P0 | DLNA subtitle extension (Phase 1) | Day 1 | At least one mainstream Quest player recognises it |
| P0 | Subtitle sniff/transcode | Day 1 | utf-8 output for srt/ass/vtt |
| P1 | mp4+srt remux cache (Phase 2) | Day 2–3 | Target clients show subtitles by default |
| P1 | LRU/TTL GC | Day 3 | Cache stays under cap |
| P1 | MKV → MP4 auto remux (Phase 3) | Day 4 | Zero mkv items in DLNA listing |
| P2 | Background pre-remux worker | Day 5 | First-play latency drops from seconds to 0 |
| P2 | Multi-language switch validation | Day 5 | 3-language toggle works |

---

## 7. Code Touch Points (locations only, no edits in this doc)

- `dlna/content_directory.py:438-510` `_didl_for()`: DIDL render entry; extend res/sec emission.
- `dlna/content_directory.py:235-330` `_video_items_from_index()`: item dict construction; inject `subtitles`.
- `http_app/routes_media.py` `/media/{name}` handler: attach `CaptionInfo.sec` headers; choose source vs cache.
- `http_app/routes_media.py` router root: register `/subs/{rel}` GET/HEAD.
- `utils/media_index.py` `IndexedChild`: extend `child.video` with `subtitles: tuple[Path, ...]` so DIDL builds skip a disk scan.
- `cache/subbed/`: new cache directory plus index file.

---

## 8. Acceptance Checklist

- [ ] BubbleUPnP DMR lists subtitle tracks and switches between them
- [ ] At least one of DeoVR / Pigasus / Skybox on Quest3 shows subtitles
- [ ] `/media/{name}` Range/HEAD behaviour unchanged
- [ ] `/subs/{rel}` Range/HEAD behaviour correct
- [ ] After mp4+srt remux cache hit, Range / progress bar accurate
- [ ] Cache GC does not delete in-flight files
- [ ] mkv files appear as mp4 in DLNA listings
- [ ] PyNv passthrough path no longer triggers Matroska stall
- [ ] Full regression: existing 8K passthrough, live chapter, alpha-live behaviour intact

---

## 9. Relation to the 8K 40 fps Plan

- The subtitle plan is decoupled from `IMPL_PLAN_8K_40FPS_20260515`; both can proceed in parallel without sharing code hot paths.
- Phase 3 (MKV→MP4 remux) is complementary to the 8K plan's MKV strategy: the 8K plan keeps "an mkv already playing from stalling", this plan eliminates "mkv at the source".
- Neither affects the PyNv three-stage pipeline, CUDA Graph, or TRT EP work tracks.
