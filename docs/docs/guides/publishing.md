---
icon: octicons/cloud-16
---

# Publishing to mjswan Cloud

[Deployment](deployment.md) covers hosting a built `dist/` yourself. mjswan Cloud is the
alternative: upload only the *data* files and get a hosted page back, with no static host
to configure and no `base_path` to get right.

!!! note "Beta"
    mjswan Cloud is in beta. The `publish` verb, the token env vars, and the URL shape are
    stable enough to script against; the hosted UI is still moving.

## What gets uploaded

Only data files travel — the compiled JavaScript never does. Cloud renders your
simulation with its own pinned copy of the engine, loaded from a CDN.

| Uploaded | Not uploaded |
|---|---|
| `manifest.json` | `index.html`, `logo.svg`, `robots.txt` |
| `scene.mjz` / `scene.mjb` | the compiled JS/CSS bundle and its WASM (`assets/`) |
| `policy/<policy-id>.onnx` | `_headers`, `coi-serviceworker.js` |
| traced graphs (`mdp/<mdp-id>/{obs,term,command,event}/`) | |
| `assets/<motion-id>.npz`, `assets/<splat-id>.spz`, `.ply` colliders | |
| `<project-id>/LICENSE`, `NOTICE` and each scene's `LICENSE.<component>`, `NOTICE.<component>` (up to 64 KB each) | the root `LICENSE` — the engine's own |

What travels is exactly the [simulation document](../getting-started/core-concepts.md#output-structure),
so a `.swn` written by `app.save_document()` publishes the same file set as the directory
it came from: `mjswan publish sim.swn`.

Limits: 50 MB per file, 200 MB total, 64 files.

Before the first byte moves, `publish` prints every license file it is about to carry with
the license it identifies — `demo/LICENSE (Apache-2.0)`,
`demo/go2/LICENSE.unitree_go2 (BSD-3-Clause)` — so nothing travels unseen. A file under a
*restricted* license (Creative Commons NC / SA / ND, the GPL family) earns a warning naming
the clause; the publish goes ahead, and the responsibility is yours. A file the platform
recognises as forbidding redistribution — the Max Planck research license behind SMPL and
AMASS, Universal Robots' graphical-documentation terms, or one tagged `LicenseRef-AMASS` /
`LicenseRef-SMPL` — is refused locally, naming the file, exactly as a custom-JS build is; the
platform refuses the same file at commit. A custom text it cannot classify is carried and
shown without comment. See [Licenses](../getting-started/core-concepts.md#licenses) for how
the files get into the build.

!!! warning "Custom-JavaScript builds are rejected"
    A build whose `manifest.json` carries `uses_custom_js: true` — one using a
    `*Binding` with `ts_src`, i.e. an author-written TypeScript term class — cannot be
    published. Cloud will not execute author-supplied code in its own origin. Traced ONNX
    terms are fine; they are inert data run by a fixed runtime. This is checked locally
    before any upload starts.

## From the CLI

```bash
python build.py                # writes dist/
mjswan publish dist --title "G1 Locomotion"
```

The first `publish` runs a GitHub sign-in automatically (a `gh`-style loopback OAuth flow)
and prints the resulting page URL:

```
Published! https://mjswan.com/s/<id>
```

Manage the session explicitly when you want to:

```bash
mjswan login             # or --no-open to print the URL instead, e.g. over SSH
mjswan whoami
mjswan logout
```

## From Python

`Builder.build()` returns an `MjswanApp`, which publishes directly — no intermediate
`dist/` path to pass around:

```python
app = builder.build()
result = app.publish(title="G1 Locomotion", tags=["locomotion", "g1"])
print(result.id)
```

## In CI

Interactive login has no place in a pipeline. Set `MJSWAN_TOKEN` instead and the browser
flow is skipped:

```yaml
- name: Build and publish
  run: |
    uv run python build.py
    uv run mjswan publish dist --title "Nightly"
  env:
    MJSWAN_TOKEN: ${{ secrets.MJSWAN_TOKEN }}
    MJSWAN_NO_LAUNCH: "1"
```

| Variable | Effect |
|---|---|
| `MJSWAN_TOKEN` | Access token. Equivalent to `--token`; skips the interactive login. |
| `MJSWAN_API_BASE` | Cloud API base URL. Defaults to `https://api.mjswan.com`. |
| `MJSWAN_WEB_BASE` | Web app base used to build the printed page URL. Defaults to `https://mjswan.com`. |

## Embedding a published simulation

A published page can be embedded like any other, and the npm engine can load a published
`manifest.json` directly. See [Embedding](embedding.md) and the
[Engine API](../api/engine.md).
