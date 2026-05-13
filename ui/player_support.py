from __future__ import annotations

import json
from dataclasses import dataclass

from ui.resources import PLAYER_SUPPORT_PATH


@dataclass(frozen=True)
class PlayerSupportRow:
    player: str
    alpha: bool
    gray_green: bool
    chroma_key: bool
    website_url: str
    notes_url: str


def load_player_support() -> list[PlayerSupportRow]:
    payload = json.loads(PLAYER_SUPPORT_PATH.read_text(encoding="utf-8-sig"))
    rows: list[PlayerSupportRow] = []
    for item in payload.get("rows", []):
        rows.append(
            PlayerSupportRow(
                player=str(item.get("player", "")),
                alpha=bool(item.get("alpha")),
                gray_green=bool(item.get("gray_green")),
                chroma_key=bool(item.get("chroma_key")),
                website_url=str(item.get("website_url", "")),
                notes_url=str(item.get("notes_url", "")),
            )
        )
    return rows
