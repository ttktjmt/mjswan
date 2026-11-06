"""A file as mjswan Cloud stores it: whole, or cut into pieces, each named by the SHA-256
of its bytes (mjswan-cloud ADR 0017 §2.6).

The rules are the platform's (its ``packages/blobs``), held to the same test files
(``tests/blob_vectors``), so a build is cut the same wherever it is published from.
"""

from __future__ import annotations

import hashlib

import numpy as np

KiB = 1024
MiB = 1024 * KiB

#: A piece longer than this is cut again, by its content.
PIECE_MAX_BYTES: int = 16 * MiB


def sha256(data: bytes | memoryview) -> str:
    """Lowercase hex, as a blob is named."""
    return hashlib.sha256(data).hexdigest()


def cut(name: str, data: bytes) -> dict:
    """``data`` as the file map records it under ``name``: ``sha256`` and ``size``, and
    ``chunks`` (``[sha256, size]`` in order) when it is more than one piece. An ``.onnx``
    is cut at its tensors, and any piece over 16 MiB by its content."""
    view = memoryview(data)
    cuts = _onnx_cuts(view) if name.lower().endswith(".onnx") else []
    bounds = [0, *cuts, len(view)]
    pieces: list[memoryview] = []
    for start, end in zip(bounds, bounds[1:]):
        piece = view[start:end]
        pieces.extend(_cdc(piece) if len(piece) > PIECE_MAX_BYTES else [piece])
    entry: dict = {"sha256": sha256(view), "size": len(view)}
    if len(pieces) > 1:
        entry["chunks"] = [[sha256(p), len(p)] for p in pieces]
    return entry


# ── ONNX ──────────────────────────────────────────────────────────────────────
# A cut where the raw data of each tensor of 32 KiB or more begins, so a policy trained
# again re-sends only the layers that changed, and one before the first `metadata_props`,
# so editing the metadata re-sends only the piece it begins. The tensors are the graph's
# initializers and its nodes' attribute tensors (`t`, `tensors`), subgraphs (`g`,
# `graphs`) included. At most 64 pieces: past that the bar rises to a 64th of the file,
# and cuts after the 63rd are dropped.
#
# Read as protobuf's wire format, descending only into those fields. A file that does not
# read that way is not cut: a length past the end of its message, a varint past ten bytes
# or past that end, a key of 2^32 or more, a wire type other than 0, 1, 2 or 5, or graphs
# nested more than 16 deep.

_TENSOR_MIN_BYTES = 32 * KiB
_ONNX_MAX_PIECES = 64
_MAX_GRAPH_DEPTH = 16


class _NotOnnx(Exception):
    pass


def _onnx_cuts(data: memoryview) -> list[int]:
    tensors: list[tuple[int, int]] = []
    metadata: int | None = None

    # ModelProto 7 graph, 14 metadata_props · GraphProto 1 node, 5 initializer · NodeProto
    # 5 attribute · AttributeProto 5 t, 6 g, 10 tensors, 11 graphs · TensorProto 9 raw_data.
    def graph(start: int, end: int, depth: int) -> None:
        if depth > _MAX_GRAPH_DEPTH:
            raise _NotOnnx
        for field, _, s, e in _fields(data, start, end):
            if field == 1:
                node(s, e, depth)
            elif field == 5:
                tensor(s, e)

    def node(start: int, end: int, depth: int) -> None:
        for field, _, s, e in _fields(data, start, end):
            if field == 5:
                attribute(s, e, depth)

    def attribute(start: int, end: int, depth: int) -> None:
        for field, _, s, e in _fields(data, start, end):
            if field in (5, 10):
                tensor(s, e)
            elif field in (6, 11):
                graph(s, e, depth + 1)

    def tensor(start: int, end: int) -> None:
        for field, _, s, e in _fields(data, start, end):
            if field == 9:
                tensors.append((s, e - s))

    try:
        for field, at, s, e in _fields(data, 0, len(data)):
            if field == 7:
                graph(s, e, 1)
            elif field == 14 and metadata is None:
                metadata = at
    except _NotOnnx:
        return []

    def cuts_from(least: int) -> list[int]:
        at = {s for s, size in tensors if size >= least}
        if metadata is not None:
            at.add(metadata)
        return sorted(a for a in at if 0 < a < len(data))

    cuts = cuts_from(_TENSOR_MIN_BYTES)
    if len(cuts) >= _ONNX_MAX_PIECES:
        cuts = cuts_from(max(_TENSOR_MIN_BYTES, -(-len(data) // _ONNX_MAX_PIECES)))
    # ponytail: dropping the last cuts is reachable only by 63 tensors of a 64th each, plus metadata.
    return cuts[: _ONNX_MAX_PIECES - 1]


def _fields(data: memoryview, start: int, end: int):
    """``(field, key at, payload start, payload end)`` for each length-delimited field of
    the message in ``[start, end)``."""
    p = start

    def varint() -> int:
        nonlocal p
        value = 0
        for n in range(11):
            if p >= end or n == 10:
                raise _NotOnnx
            b = data[p]
            p += 1
            value |= (b & 0x7F) << (7 * n)
            if b < 0x80:
                return value
        raise _NotOnnx  # unreachable

    while p < end:
        at = p
        key = varint()
        if key > 0xFFFFFFFF:
            raise _NotOnnx
        wire = key & 7
        if wire == 0:
            varint()
        elif wire == 1:
            p += 8
        elif wire == 5:
            p += 4
        elif wire == 2:
            length = varint()
            if length > end - p:
                raise _NotOnnx
            yield key >> 3, at, p, p + length
            p += length
        else:
            raise _NotOnnx
    if p > end:
        raise _NotOnnx


# ── Content-defined chunking ──────────────────────────────────────────────────
# For a piece over 16 MiB. A cut falls after a byte where a gear hash of the 32 bytes
# ending there has its top 20 bits zero, at least 256 KiB after the last cut, and 4 MiB
# after it at the latest. Where it falls depends on those 32 bytes alone, so bytes
# inserted or removed move only the cuts near them.

_CDC_MIN_BYTES = 256 * KiB
_CDC_MAX_BYTES = 4 * MiB

#: For byte i: the first four bytes, big-endian, of the SHA-256 of the single byte i.
_GEAR = np.array(
    [
        int.from_bytes(hashlib.sha256(bytes([i])).digest()[:4], "big")
        for i in range(256)
    ],
    dtype=np.uint32,
)


def _cdc(data: memoryview) -> list[memoryview]:
    buf = np.frombuffer(data, dtype=np.uint8)
    pieces: list[memoryview] = []
    start = 0
    while start < len(buf):
        last = min(len(buf), start + _CDC_MAX_BYTES)
        end = last
        # The hash at byte i is sum(GEAR[b[i-k]] << k for k < 32) on 32 bits, computed for
        # every byte where a cut may fall at once.
        first = start + _CDC_MIN_BYTES - 32
        if first + 31 < last:
            gear = _GEAR[buf[first:last]]
            h = np.zeros(len(gear) - 31, dtype=np.uint32)
            for k in range(32):
                h += gear[31 - k : len(gear) - k] << np.uint32(k)
            hits = np.flatnonzero((h >> np.uint32(12)) == 0)
            if hits.size:
                end = first + 32 + int(hits[0])
        pieces.append(data[start:end])
        start = end
    return pieces


__all__ = ["PIECE_MAX_BYTES", "cut", "sha256"]
