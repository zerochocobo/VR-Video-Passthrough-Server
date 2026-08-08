# DLNA “All Videos” Development Plan (2026-07-21)

## Goals

1. Keep the Settings page height consistent with the Home page so navigation does not resize the window.
2. Add a persisted, default-off “Show ‘All Videos’ on DLNA home” switch to the DLNA settings group.
3. Pass the current UI language to the server; when enabled, scan all configured video roots at startup and cache supported videos in name order.
4. Append a localized “All Videos” container to the DLNA home. Its children are filename containers; entering one exposes the source item and all enabled realtime playback entries, including VR-renamed titles.

## Steps

1. Align page-size handling with the Home page and remove the Settings-specific height difference.
2. Extend `ui.settings.Settings` defaults, environment export, settings controls, and translations.
3. Add a startup media index in the DLNA content-directory module: walk `VIDEO_DIRS`, filter `VIDEO_EXTS`, sort by display name, and retain safe paths plus VR display names.
4. Add ObjectID encoding/decoding, DIDL generation, and Browse dispatch for the virtual All Videos root and filename containers, reusing existing source/live/SI/RM/VR naming logic.
5. Build the index during server startup only when enabled, and invalidate related caches when configuration or roots change.
6. Add tests for default-off behavior, filtering/sorting, multi-root safety, root ordering, filename-container children, and localized titles.
7. Run syntax checks and tests, then record results in the handover document.

## Acceptance

- Switching between Home and Settings does not change window height.
- The new switch is unchecked by default and translated in all three UI languages.
- When enabled, startup builds the index and DLNA home ends with localized “All Videos”.
- The first level contains only filename containers; the next level contains the source and enabled realtime entries, with VR rename applied.
- Disabled mode preserves existing behavior and skips the full scan.
