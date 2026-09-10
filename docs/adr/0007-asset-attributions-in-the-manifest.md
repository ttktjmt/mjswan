# Asset attributions in the manifest — declared where the model is known

> Status: **Proposed** — adds optional keys to the `format: 1` manifest of
> [ADR 0006](0006-swn-simulation-document.md) (§7: a reader ignores keys it does
> not know, so `format` stays 1) and gives `publish` a second local UX gate
> beside `uses_custom_js` ([ADR 0003](0003-declarative-mdp-terms-alongside-custom-js.md)).
> The platform half is mjswan Cloud ADR 0013, which this record is written to
> match; the shapes below are the contract between the two.

## Context

A build packs a model as `scene.mjz`: the MJCF plus every mesh and texture it
names (`to_zip_deflated`). The `LICENSE` that sat beside the model on disk is not
named by the XML, so it is not packed. The `.swn` document carries any loose
file a script copies under a project directory (`document_files` takes the
whole tree), but `publish` selects files by extension (`DATA_EXTENSIONS`) and
the platform's validator does the same, so an extension-less `LICENSE`, a
`NOTICE` or a `README.md` never reaches mjswan Cloud, and a `.swn` that has them
publishes as one that does not.

The models mjswan is used with mostly require exactly that file to travel. Every
Unitree model in `mujoco_menagerie` is BSD-3-Clause and asks that binary
redistributions *"reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials
provided with the distribution"*; Apache-2.0 asks for the retained notices and
the `NOTICE` contents; MIT for the notice. Menagerie is otherwise uniformly
permissive (59 models: Apache-2.0, BSD-3-Clause, MIT, BSD-2-Clause,
BSD-3-Clause-Clear), as are `myo_sim`, MyoSuite, `mujoco_playground` and mjlab.
What is not permissive is some of the **motion data** a tracking task bundles:
LAFAN1 and Unitree's retargeted LAFAN1 are CC BY-NC-ND 4.0, and AMASS-derived
clips may not be made available to third parties at all — `_write_scene_motions`
copies those `.npz` files into the build all the same.

The one place that knows what a model *is* is the build. `spec.modelfiledir` is
the directory the MJCF was loaded from (`collect_spec_assets` already reads it);
the meshes resolve to directories on disk; `add_scene_mjlab` knows the task id.
None of that survives into the bytes the platform receives. So the declaration
has to be made here, and the manifest — the document's one descriptor — is
where a statement about the document belongs.

The demo already tries: `examples/demo/main.py::_copy_licenses` copies `LICENSE`
and `NOTICE` into scene directories. For a menagerie-hosted description it
copies `REPOSITORY_PATH/LICENSE`, which is the *repository's* Apache-2.0 (Google
DeepMind) rather than the *model's* BSD-3-Clause one directory down at
`PACKAGE_PATH/LICENSE`. A file copy is the wrong primitive; a declaration with a
detector behind it is the right one.

## Decision

### 1. Two optional manifest keys; `format` stays 1

```json
{
  "format": 1,
  "version": "0.10.0",
  "license": "Apache-2.0",
  "projects": [{ "id": "demo", "name": "Demo", "scenes": [{
    "id": "go2", "name": "Go2", "scene": "scene.mjz",
    "attributions": [{
      "name": "Unitree Go2", "kind": "model", "spdx": "BSD-3-Clause",
      "copyright": "Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. (\"Unitree Robotics\")",
      "url": "https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2"
    }],
    "mdps": [], "policies": []
  }]}]
}
```

- **`license`** (top level, optional): an SPDX license expression for **the
  work** — what the author lets others do with the simulation. Absent means
  all rights reserved.
- **`attributions`** (per scene, optional): notices for **third-party
  components** the scene contains, one entry each.

```ts
export interface ManifestAttribution {
  name: string;         // "Unitree Go2"
  spdx: string;         // an SPDX id, or "LicenseRef-<id>" for a license SPDX does not list
  copyright?: string;   // the holder's line, verbatim
  url?: string;         // where the component lives
  notice?: string;      // NOTICE contents; the full text when spdx is a LicenseRef
  kind?: 'model' | 'motion' | 'policy' | 'splat' | 'texture' | 'other';
}
```

Per scene rather than per document because the scene is where the builder knows
what a model is, and a multi-robot document has different notices per scene. A
consumer that wants one list flattens and de-duplicates on `(spdx, copyright)`,
which is what mjswan Cloud does for its row. `parseManifest` types the keys and
otherwise ignores them; the engine has no use for a notice.

For a listed SPDX id the `notice` carries only what the license requires to be
*reproduced* — the copyright line, any `NOTICE` file — not the license text,
which the id identifies. The full text is stored only for `LicenseRef-*`, where
nothing else identifies it. A BSD attribution is therefore ~300 bytes, not
1.5 KiB, and an Apache one is the size of its `NOTICE`.

### 2. Detection at `add_scene`, in three layers

`ProjectHandle.add_scene` (and `add_scene_mjlab`, which calls it) fills the
scene's attributions when the author has not, de-duplicating on `(spdx,
copyright)`:

1. **Files beside the model.** Candidate directories are `spec.modelfiledir` and
   the directories the spec's meshes, textures, hfields and skins resolve to
   (the same resolution `collect_spec_assets` performs), each with up to two
   parents. In each, `LICENSE*`, `COPYING*` and `NOTICE*` are read. The SPDX id
   comes from a header match on the first lines — BSD-3-Clause, BSD-2-Clause,
   Apache-2.0, MIT, the CC BY family — and a file the matcher cannot classify
   becomes `LicenseRef-<dirname>` with its text as the `notice`. This finds
   menagerie's per-model `LICENSE` whether the model was opened directly, via
   `robot_descriptions` (`PACKAGE_PATH` *is* the model directory) or via
   `mujoco_playground` (the XML lives in playground; the meshes resolve into
   `external_deps/mujoco_menagerie/<model>/assets/`, one level below the
   `LICENSE`). Two parents, not more: a third would reach a repository root
   and attribute the wrong license — the demo's failure mode.
2. **A table of known assets** (`mjswan/licenses.py`): `{name, spdx,
   copyright, url}` keyed by menagerie directory name, `spec.modelname` and
   mjlab task id, for the models mjswan is actually used with. This covers
   mjlab's asset zoo, whose robot XMLs ship with no license file beside them,
   and it is what the CLI *recommends* when it recognises a model that carries
   no declaration.
3. **The author.** `SceneHandle.add_attribution(name, spdx, *, copyright=None,
   url=None, notice=None, kind="model")`, `SceneHandle.clear_attributions()`
   for a detection that is wrong, `MotionConfig(..., attribution=...)`,
   `PolicyConfig(..., attribution=...)`, and `Builder(license=...)` /
   `Builder.set_license()` for the work. Motion clips and policy weights are
   only ever declared this way: there is no file on disk to find, and a
   detector that guessed would be worse than one that stays silent.

Detection runs at `add_scene` rather than in `_save_web` because the spec is
dropped after packaging (`scene.spec = None`) and because the handle is where an
author can inspect and correct what was found before the build.

### 3. `publish` prints, warns, refuses

`plan_publish` reads the manifest's declaration and:

- **prints it** — `Attributions: Unitree Go2 (BSD-3-Clause) · …` and `License:
  Apache-2.0` (or `all rights reserved`) — so a publish never declares something
  the author did not see;
- **warns** for a *restricted* license (the CC NC / SA / ND family, GPL-family,
  any `LicenseRef-` not known to be blocked), naming the component and the
  clause, before any network call;
- **refuses** a *blocked* license (`LicenseRef-AMASS`, `LicenseRef-SMPL`,
  `LicenseRef-UR-Graphical-Documentation`: sources whose terms forbid making the
  data available to third parties), as a `PublishError` on `manifest.json` with
  the same shape as the custom-JS refusal, before any network call. The
  platform refuses the same declaration server-side; this is the fast local
  copy of that gate, exactly as `uses_custom_js` is.

`publish_dist(..., license=None, attributions=None)` and `mjswan publish
--license` override the manifest's `license`; an explicit `attributions` list
replaces the manifest's. When neither is given the request body carries nothing
new — the platform reads the declaration from the `config` it already receives,
so a platform that predates this ADR ignores the keys and a CLI that predates
the platform change keeps working.

The tier table is a small module, `mjswan/licenses.py`, with the same contents
as the platform's `lib/licenses.ts`. The two are kept in step by hand, like
`name2id_cases.json`; the platform's copy is authoritative for what a publish
is accepted with.

### 4. `mjswan info` lists the declaration

Beside the projects, scenes and policies it already prints. The standalone SPA
does **not** yet render notices from the manifest; that is the natural follow-up
(a GitHub Pages deployment of a Unitree demo has the same obligation the
platform has), and it is out of scope here.

## Considered options

| Option | Why not |
|---|---|
| Copy `LICENSE` files into the scene directory and upload them | The platform's extension allowlist and pinned `Content-Type` exclude them by design; `scenes/{id}/` is immutable, so a correction would need a new mutable object; and the demo shows the copy picking the repository license over the model's. |
| Put the notice inside `scene.mjz` | Invisible to every reader — the engine opens the archive as a model — and it makes the model archive carry something the model does not need. |
| One `attributions` list at the document root | Loses which scene a notice belongs to in a multi-robot document; flattening is the consumer's one-liner, un-flattening is not possible. |
| A full license-identification library | The sources mjswan is used with either carry a `LICENSE` beside the model or are in the known-assets table; a header matcher plus that table is the whole problem. |
| Detect in `_save_web` | The spec and its directories are gone by then, and the author has no chance to correct what was found. |

## Consequences

- A build from a menagerie, `robot_descriptions` or `mujoco_playground` model
  carries the model's notice with no author action; an mjlab build carries it
  from the table; a build from an unknown model carries nothing and says so.
- `manifest.json` grows by the notices — hundreds of bytes per model for the
  listed licenses, the license text for a `LicenseRef`. Readers that do not
  know the keys ignore them; `format` stays 1.
- A `.swn` carries the declaration wherever it goes; a self-hosted `dist/` does
  too, for a future SPA that renders it.
- `examples/demo/main.py::_copy_licenses` is retired: detection finds the
  correct per-model `LICENSE`, and the notice reaches the platform through the
  manifest instead of a file the platform refuses.
- `publish` gains a second local gate. Like the custom-JS gate it can only act
  on what the document declares, and that is its job: a declared
  non-redistributable asset is refused rather than uploaded and refused.
- `docs/getting-started/core-concepts.md` (output structure, the manifest
  shape), `docs/guides/publishing.md` (what is declared, `--license`, the
  warning and refusal), the CHANGELOG under Unreleased, and `manifest.d.ts` /
  `src/manifest/index.ts` (the `ManifestAttribution` type) change with it.

## Phased execution plan

1. `mjswan/licenses.py`: SPDX allowlist, tiers, header matcher, known-assets
   table; unit tests with fixture license texts.
2. Manifest keys: `SceneConfig.attributions`, `Builder.license`, `_scene_entry`
   and `_save_manifest` emit them when set; `ManifestAttribution` in
   `src/manifest/index.ts`; `parseManifest` unchanged in behaviour; tests via
   `build_manifest`.
3. Detection at `add_scene` / `add_scene_mjlab`: candidate directories, the
   two-parent bound, de-duplication; tests against a temporary directory laid
   out like a menagerie model and like a playground checkout.
4. Author API: `add_attribution`, `clear_attributions`, `MotionConfig` /
   `PolicyConfig.attribution`, `Builder(license=)` / `set_license`.
5. `publish`: print, warn, refuse; `--license`; explicit overrides; tests
   beside `TestPlanPublish` and `TestPublishDist`.
6. `mjswan info`; retire the demo's `_copy_licenses`; docs and CHANGELOG.

## Acceptance criteria

- [ ] A scene loaded from a directory holding a BSD-3-Clause `LICENSE` gets one
      attribution with `spdx: "BSD-3-Clause"` and the copyright line, and
      nothing else.
- [ ] A model whose XML lives outside the menagerie checkout but whose meshes
      resolve into it (playground layout) is attributed to the model, not to
      the repository.
- [ ] A directory two levels above a repository root does not contribute a
      license.
- [ ] `add_scene_mjlab` for a Unitree G1 task attributes Unitree G1 from the
      table.
- [ ] `format` is still 1, `parseManifest` accepts a document with the keys,
      and an engine build that predates them renders it unchanged.
- [ ] `plan_publish` refuses a `LicenseRef-AMASS` declaration before any
      network call, with the file named, and warns for `CC-BY-NC-ND-4.0`.
- [ ] `--license` reaches the request body and overrides the manifest's
      `license`; with no flag the body carries no `license` key.
- [ ] A `.swn` round-trips the keys byte for byte.
