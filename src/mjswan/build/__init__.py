"""What ``Builder.build()`` does: turn the configured projects into a document on disk.

:mod:`.pipeline` drives it; :mod:`.frontend` builds the SPA the document is laid under;
:mod:`.mjz` packs a scene; :mod:`.asset` copies its clips and splats; :mod:`.mdp` traces
its terms to ``.onnx`` graphs; :mod:`.manifest` assembles the entries and writes
``manifest.json``. Nothing here is imported by the object model at import time: the
root package stays light, and the tracer's torch is paid for only by a build that
traces.
"""
