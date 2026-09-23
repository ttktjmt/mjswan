"""What ``Builder.build()`` does: turn the configured projects into a document on disk.

:mod:`.pipeline` drives it; :mod:`.frontend` builds the SPA the document is laid under;
:mod:`.mjz` packs a scene; :mod:`.asset` copies its clips and splats; :mod:`.mdp` traces
its terms to ``.onnx`` graphs; :mod:`.manifest` assembles the entries and writes
``manifest.json``. Only :mod:`.mjz` is imported with the object model; the rest loads
when a build runs, and the tracer (:mod:`mjswan.compile`) only when a term is traced.
"""
