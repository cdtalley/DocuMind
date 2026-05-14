"""Indexed corpus selector: public (e.g. Wikipedia-scale) vs papers (PDFs / arXiv / legacy bundle)."""

from __future__ import annotations

from typing import Literal

LibraryId = Literal["public", "papers"]
