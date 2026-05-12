# Tests

Run the lightweight local test suite from the repository root:

```powershell
uv run python -m unittest discover -s tests
```

The suite avoids starting the real DLNA server or running GPU conversion. It
checks UI construction, i18n key parity, settings-to-environment mapping,
ContentDirectory passthrough modes, log rotation, and the formal offline CLI
argument plumbing.
