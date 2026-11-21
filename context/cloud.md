# mjswan Cloud

`cloud/`: `mjswan publish` and the `mjswan login` session. `cloud/` reads `document/` and `license/` and never imports `build/` ([package-layout.md](package-layout.md)). The platform lives in its own repository; this file holds only what the mjswan side needs. Where `cloud/` copies a platform rule (the limits, the cutter, what is refused), the platform is the source of truth and the copy is kept in step by hand.

## What `publish` sends

`publish.py`'s `publish_dist()` takes a built directory or a `.swn` (unpacked locally through `document.as_directory`, so both send the same files) and sends only the document's data files: `.json`, `.mjz`, `.onnx`, `.npz`, `.ply`, `.spz`, the skins' `.vrm` / `.glb`, and license files admitted by name. Never the engine (JS, CSS, WASM, `index.html`, `assets/`): Cloud supplies its own, chosen by the document's `version` ([document.md](document.md)). A symlink is refused. Files go one by one, never as one `.swn`.

Files are content-addressed: `blob.py` cuts each into pieces by the platform's rules, pinned by `tests/blob_vectors`, and `publish` uploads only the pieces Cloud does not already hold, then commits. Change `blob.py` only together with the platform.

The size and count limits (`MAX_*` in `publish.py`) are checked before any upload, so an oversized build fails locally.

## What `publish` refuses, before any network call

- **A build with `uses_custom_js: true`**: Cloud runs no author code ([frontend.md](frontend.md)). A term that needs TypeScript reaches Cloud through an mjswan change that lets it trace, released, and the document rebuilt against that release.
- **A `.mjb` scene** (`add_scene(model=...)`): `.mjb` loads only in the MuJoCo that wrote it; build it with `add_scene(spec=...)`.
- **A license that forbids redistribution**, per file; a restricted one is sent with a warning ([licenses.md](licenses.md)). License files are admitted only at `<project-id>/LICENSE` / `NOTICE`, `<project-id>/licenses/LICENSE.<component>` / `NOTICE.<component>`, or a scene directory's; never the root `LICENSE`, which is the engine's.
- **A skin Cloud would refuse**: a file the engine cannot draw, or a VRM whose license forbids redistribution. A VRM only its author (or someone they licensed) may use is sent only once `attest` says the publisher may ([skins.md](skins.md)).

`mjswan info` prints the same declarations for a directory or a `.swn`.

## Login

`mjswan login` (`auth.py`) signs in through the browser and keeps a refreshable session in `~/.config/mjswan/credentials.json` (mode `0600`), so the CLI never holds a long-lived secret; the client key in `auth.py` is public by design. `MJSWAN_TOKEN` skips the sign-in for CI. `transport.py` sets the user agent every request sends, since the API refuses Python's default one.
