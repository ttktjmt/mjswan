# mjswan Cloud

`cloud/`: publishing a built document to mjswan Cloud, and the `mjswan login` session. `cloud/` reads `document/` and `license/` and never imports `build/` ([package-layout.md](package-layout.md)). The platform itself lives in the mjswan-cloud repository; several rules here mirror its own and are kept in step by hand.

## What is sent: data, never the engine

`publish.py`'s `publish_dist()` takes a built directory or a `.swn` (unpacked locally through `document.as_directory`, so both upload the same file set) and publishes only *data* files: `.json`, `.mjz`, `.onnx`, `.npz`, `.ply`, `.spz`, the skins' `.vrm` / `.glb`, and license files admitted by name. Never the compiled JS, CSS, WASM or `index.html`: Cloud loads a pinned engine from its own CDN, picked by the document's `version` ([document.md](document.md)). The SPA's own `assets/` directory is skipped, and a symlink is refused.

Cloud receives individual files, not a `.swn`: one archive would need a server-side unpack, lose each file's content type, put every document under one file's size limit, and forfeit sending only what changed.

Limits, mirroring the server's so a publish fails fast and locally: 50 MB a file or `.mjz` entry, 128 MB total with scenes unpacked and a piece two files share counted once, 200 files, 1,000 pieces.

## How it is sent: content-addressed pieces

Cloud keeps each file once, by its SHA-256. `publish` sends a file map (each `.mjz` as its entries, each file as the pieces `blob.py` cuts it into, each license file as the hash of its text), `PUT`s only the pieces Cloud asks for, calls `verify` until Cloud has checked every piece's SHA-256, then commits, re-sending once any piece a takedown released meanwhile.

`blob.py` is the cutter: an `.onnx` is cut before every tensor of 32 KiB or more and before its metadata, and any piece over 16 MiB again by content-defined chunking. The rules are the platform's, held to its own test files (`tests/blob_vectors`), so a build is cut the same wherever it is published from.

## What is refused, before any network call

- **A build with `uses_custom_js: true`.** Cloud renders untrusted documents in one shared engine, so it accepts only builds with no author code; the reasoning is in [frontend.md](frontend.md). The way onto Cloud for a term that needs TypeScript is an mjswan change that lets it trace (a slot the browser's reader serves, say), released, and the document built against that release.
- **A `.mjb` scene** (an `add_scene(model=...)` scene): `.mjb` is not a data extension and loads only in the MuJoCo that wrote it; build it with `add_scene(spec=...)`.
- **A license that forbids redistribution**, refused per file; a restricted one is published with a warning; each license file is printed with its identified license ([licenses.md](licenses.md)). License files are admitted only at `<project-id>/LICENSE` / `NOTICE`, `<project-id>/licenses/LICENSE.<component>` / `NOTICE.<component>`, or a scene directory's, as earlier builds wrote them; never the root `LICENSE`, which is the engine's.
- **A skin Cloud would refuse at commit**: each is checked as Cloud checks it (the file, its extension, and a VRM's license and meta, read by `license/vrm.py`). A VRM whose license forbids redistribution is refused, since hosting it is redistributing it; a VRM only its author (or someone they licensed) may use is sent only once `attest` says the publisher may, and the commit lists it in `attested_skins`; a non-commercial VRM publishes with a warning ([skins.md](skins.md)).

`mjswan info` prints the same declarations for a directory or a `.swn`.

## `auth.py` and `transport.py`

`auth.py` is a `gh`-style loopback PKCE OAuth flow against Supabase's GitHub OAuth: it opens the browser, captures the session on a transient `127.0.0.1` listener, and persists the access token and the rotating refresh token in `~/.config/mjswan/credentials.json` (mode `0600`), refreshing the short-lived access token transparently. The CLI never holds a long-lived secret; the Supabase publishable key embedded there is public. `MJSWAN_TOKEN` bypasses the flow for CI.

`transport.py` holds the user agent both clients send: Cloudflare, in front of the API, rejects the stdlib's default `Python-urllib` agent with a 403.
