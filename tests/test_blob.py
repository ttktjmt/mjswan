"""The cutter against the test files mjswan Cloud's cutter shares (tests/blob_vectors).

Layer: L1 (pure Python).
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from mjswan.cloud.blob import cut

VECTORS = Path(__file__).parent / "blob_vectors"


@cache
def _xorshift32(seed: int, size: int) -> bytes:
    """Each step's state, four bytes little-endian (blob_vectors/README.md)."""
    words = np.empty(-(-size // 4), dtype="<u4")
    x = seed
    for i in range(len(words)):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        words[i] = x
    return words.tobytes()[:size]


def _input(parts: list[dict]) -> bytes:
    return b"".join(
        (VECTORS / p["file"]).read_bytes()
        if "file" in p
        else _xorshift32(p["xorshift32"], p["size"])
        for p in parts
    )


_CASES = json.loads((VECTORS / "vectors.json").read_text())


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_cut_matches_the_shared_test_files(case):
    assert cut(case["name"], _input(case["input"])) == case["expect"]
