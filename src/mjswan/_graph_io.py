"""Where a traced MDP graph lands in the bundle, and the one write that puts it there.

The ref the manifest carries *is* the path on disk, resolved against the scene directory.
Kept free of the tracer, and so of torch, since both ends of a build import it.
"""

from __future__ import annotations

from pathlib import Path


def onnx_ref(kind: str, name: str, scope: str | None = None) -> str:
    """Scene-relative path for a traced term's ``.onnx`` file.

    *scope* is the owning MDP's directory, ``mdp/<mdp-id>``: term and group names are
    unique within an MDP but not within a scene, so unscoped, two MDPs would collide.
    """
    ref = f"{kind}/{name}.onnx"
    return ref if scope is None else f"{scope}/{ref}"


def stamp_provenance(onnx_bytes: bytes, meta: dict[str, str]) -> bytes:
    """Write who traced a graph into the model itself.

    A `.onnx` travels alone — into Netron, into mjswan Cloud's inspector — where the
    manifest that names it is out of reach. `producer_name`, `doc_string` and
    `metadata_props` (`mjswan.<key>`) are the fields every ONNX viewer already shows.
    """
    import onnx

    from . import __version__

    model = onnx.load_from_string(onnx_bytes)
    model.producer_name = "mjswan"
    model.producer_version = __version__
    kind, term, func = meta.get("kind"), meta.get("term"), meta.get("func")
    model.doc_string = f"mjswan {kind} graph {term!r}" + (
        f", traced from {func}" if func else ""
    )
    del model.metadata_props[:]
    for key, value in meta.items():
        prop = model.metadata_props.add()
        prop.key, prop.value = f"mjswan.{key}", value
    return model.SerializeToString()


def write_onnx(
    out_dir: Path, ref: str, onnx_bytes: bytes, *, meta: dict[str, str] | None = None
) -> None:
    """Write a traced graph, refusing to replace a different graph already at *ref*.

    A build wipes its output first, so an existing file is this build's: identical bytes
    are one term traced twice, different bytes mean two owners resolved to one path.
    *meta* is stamped into the model first (:func:`stamp_provenance`), so that
    comparison sees what lands on disk.
    """
    if meta:
        onnx_bytes = stamp_provenance(onnx_bytes, meta)
    path = out_dir / ref
    if path.is_file() and path.read_bytes() != onnx_bytes:
        raise ValueError(
            f"Two different graphs both want {ref!r} under {out_dir}. One path cannot "
            "carry both: the second write wins and every config still pointing at the "
            "first would load the wrong graph, with no error at playback."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_bytes)


__all__ = ["onnx_ref", "stamp_provenance", "write_onnx"]
