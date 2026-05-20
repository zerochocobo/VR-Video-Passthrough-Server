# PT DLNA Client Simulator

Local DLNA client simulator for PTMediaServer realtime playback testing.

It browses the server ContentDirectory through SOAP, shows folders/items in a
tree, and pulls selected media URLs without opening a video player. The app is a
small local Web UI served by Python and opened in your browser. For MPEG-TS live
streams it estimates video FPS by counting video PES starts in the TS stream.
This avoids local GPU decoding load while still exercising the same HTTP/DLNA
path used by real clients.

## Run

Start PTMediaServer first, then run:

```bat
python app.py
```

The tool opens:

```text
http://127.0.0.1:8765/
```

Default server URL is:

```text
http://127.0.0.1:8200
```

Use **Refresh** to browse the root directory. Expand folders or live chapter
containers on the left, select a playable item, then click **Start Pull**.

Use **Close App** in the top-right corner of the Web UI to stop the simulator.
On Windows, the packaged EXE also shows a small local control window with
**Open Web UI** and **Close App** buttons, so closing the browser tab alone is
not required to stop the background test server.

## Build EXE

From this repository's normal `uv` environment, run:

```bat
build_exe.bat
```

If you do not use `uv`, install PyInstaller into the Python environment used for
this tool and run the same command:

```bat
pip install pyinstaller
```

The executable is written under:

```text
dist\PT_DLNA_Client_Simulator.exe
```

## Notes

- No video frames are decoded or rendered.
- The browser only renders the control UI; stream pulling and FPS counting run
  inside Python.
- FPS is available for MPEG-TS streams, especially `/passthrough_live/...`.
- Raw MP4 file pulls report throughput and bytes but usually cannot report FPS
  without decoding.
- Client profiles change only HTTP headers/User-Agent so the server can take
  the same compatibility path as real players.
