from __future__ import annotations

import unittest


class PyMp4CompatTests(unittest.TestCase):
    """Guard the pymp4/construct-2.8.8-on-Python-3.10+ build path.

    Self-contained (no external media / ffmpeg): a minimal ftyp box must
    parse *and* build (serialise) byte-identically through the compat shim.
    Without the shim Box.build raises AttributeError on collections.Sequence.
    """

    def test_box_parse_and_build_roundtrip_through_shim(self) -> None:
        from utils.pymp4_compat import Box

        # size=0x14 (20B): 'ftyp' + major 'isom' + minor 0 + brand 'isom'
        raw = bytes.fromhex("00000014") + b"ftyp" + b"isom" + bytes(4) + b"isom"
        box = Box.parse(raw)
        self.assertEqual(box.type, b"ftyp")
        rebuilt = Box.build(box)
        self.assertEqual(rebuilt, raw)

    def test_container_box_build_works(self) -> None:
        """A container box (free inside a parsed structure) must also build."""
        from utils.pymp4_compat import Box

        raw = bytes.fromhex("0000000c") + b"free" + bytes(4)  # 12B free box
        box = Box.parse(raw)
        self.assertEqual(Box.build(box), raw)


if __name__ == "__main__":
    unittest.main()
