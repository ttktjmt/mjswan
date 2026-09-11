# License files in the build — `LICENSE` / `NOTICE` per project and per scene component

> Status: **Accepted** — adds two file kinds to the `format: 1` layout of
> [ADR 0006](0006-swn-simulation-document.md) without touching the manifest: a
> file the manifest does not name is not a break, since §7's reader rule is about
> keys, and these are files no engine opens. Gives `publish` a second local UX
> gate beside `uses_custom_js` ([ADR 0003](0003-declarative-mdp-terms-alongside-custom-js.md)).
> The platform half is mjswan Cloud ADR 0014; the naming rule in §1 is the
> contract between the two. Implemented in `mjswan/licenses/`, `publish.py`,
> `project.py`, `scene.py`, `builder.py` and `_cli.py`, with `tests/test_licenses.py`.

## Context

A build packs a model as `scene.mjz`: the MJCF plus every mesh and texture it
names (`to_zip_deflated`). The `LICENSE` that sat beside the model on disk is not
named by the XML, so it is not packed. The `.swn` document carries any loose
file under a project directory (`document_files` takes the whole tree), but
`publish` selects files by extension (`DATA_EXTENSIONS`) and the platform's
validator does the same, so an extension-less `LICENSE` or `NOTICE` never
reaches mjswan Cloud, and a `.swn` that has them publishes as one that does not.

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
None of that survives into the bytes the platform receives, so the declaration
has to be made here — and review settled that it is made as **files**, in the
tree the build already writes, not as manifest keys: the notice then sits
beside the bytes it covers, the upstream text travels verbatim, and the
platform's upload form handles files already.

Two things the build does today shape the layout. `install_spa` copies the
*engine's* Apache-2.0 `LICENSE` to the root of every `dist/`; that is the
engine's obligation and must stay, so the root is not available for the work's
license. And `examples/demo/main.py::_copy_licenses` already copies `LICENSE`
files into scene directories — from the wrong place: for a menagerie-hosted
description it takes `REPOSITORY_PATH/LICENSE`, the *repository's* Apache-2.0,
rather than the model's BSD-3-Clause `LICENSE` one directory down at
`PACKAGE_PATH/LICENSE`.

## Decision

### 1. Two file kinds in the layout; the manifest is unchanged

```
dist/
├── manifest.json
├── LICENSE                    ← the engine's Apache-2.0, as today; never published
├── assets/                    ← the engine; never published
└── <project-id>/
    ├── LICENSE                ← ① the work's license
    ├── NOTICE                 ← ① the work's notice
    └── <scene-id>/
        ├── scene.mjz · mdp/ · policy/ · assets/
        ├── LICENSE.<component>   ← ② one third-party component this scene contains
        └── NOTICE.<component>    ← ② its notice, if it has one
```

- **① The work's license** is per project, in the project directory. A
  multi-project document may license each project differently; mjswan Cloud
  renders `projects[0]` and reads that project's files.
- **② A third-party component** is one file per component in the scene that
  contains it: `LICENSE.unitree_go2`, `NOTICE.lafan1`. The suffix is the
  component's label; a bare `LICENSE` / `NOTICE` at scene level is accepted
  and labelled with the scene id.
- **Naming rule**, shared with the platform's validator: basename
  `^(LICENSE|NOTICE)(\.[A-Za-z0-9_-]{1,64})?$`, at depth 1 (`<project>/`) or
  depth 2 (`<project>/<scene>/`) only. The root is never a license location
  for `publish`.
- **The SPDX tag line.** A file mjswan *generates* — from a template, or from
  the known-assets table for a model that ships without one — begins with the
  SPDX short-form tag on its first line: `SPDX-License-Identifier: BSD-3-Clause`.
  A file mjswan *copies* from beside a model is copied verbatim, with no line
  added: the platform recognises the standard texts by their headers, and the
  tag is for the cases a header cannot settle (a custom or blocked license).
- `manifest.json`, `format`, and `parseManifest` are unchanged. `document_files`
  already includes these files in the `.swn`.

### 2. Detection at `add_scene` writes `LICENSE.<component>`

`ProjectHandle.add_scene` (and `add_scene_mjlab`, which calls it) records
attributions on the scene, and `_save_web` writes them as files when the scene
directory is created:

1. **Files beside the model.** Candidate directories are `spec.modelfiledir` and
   the directories the spec's meshes, textures, hfields and skins resolve to
   (the resolution `collect_spec_assets` performs). Each is searched, then its
   parents up to two levels, and the **nearest level that has any** `LICENSE*`,
   `COPYING*` or `NOTICE*` wins for that directory: the files are taken verbatim
   and become `LICENSE.<dirname>` / `NOTICE.<dirname>`, de-duplicated by
   content. This finds menagerie's per-model `LICENSE` whether the model was
   opened directly, via `robot_descriptions` (`PACKAGE_PATH` *is* the model
   directory) or via `mujoco_playground` (the XML lives in playground; the
   meshes resolve into `external_deps/mujoco_menagerie/<model>/assets/`, one
   level below the `LICENSE`, and the search stops there rather than climbing
   on to the repository's Apache-2.0). Two parents, not more: a third reaches a
   repository root and copies the wrong license — the demo's failure mode. A
   spec parsed from a string has no directory and is not searched: resolving
   its asset paths against the working directory would find whatever happens
   to be there.
2. **A table of known assets** (`mjswan/licenses/`): `{component, spdx,
   holder, url}` matched by the tokens of a menagerie directory name,
   `spec.modelname` or mjlab task id (`Mjlab-Velocity-Flat-Unitree-G1`), for the
   models mjswan is actually used with. When no file was found but the table
   knows the model, mjswan generates `LICENSE.<component>` from the license
   template with the holder's line and the SPDX tag. This covers mjlab's asset
   zoo, whose robot XMLs ship with no license file beside them.
3. **The author.** `SceneHandle.add_attribution(component, *, license=<spdx or
   path>, notice=<path or text>, copyright=None)` and
   `SceneHandle.clear_attributions()` for a detection that is wrong. Motion clips
   and policy weights are only ever declared this way: there is no file on disk
   to find, and a detector that guessed would be worse than one that stays
   silent.

The work's own files come from `Builder(license=<spdx or path>, copyright=…)`
as the default for every project, `add_project(name, license=…, copyright=…)`
per project, and `ProjectHandle.set_license()` / `set_notice()`. An SPDX id
produces the standard text from a bundled template, tagged; a path is copied.

### 3. `publish` admits the names and gates locally

`plan_publish` admits a file whose path satisfies the naming rule, beside the
extension allowlist, with `text/plain; charset=utf-8` as its advisory content
type. It then reads each admitted `LICENSE*` (they are small; the platform caps
them at 64 KiB) and:

- **prints** the declaration — `demo/LICENSE (Apache-2.0)`, `demo/go2/
  LICENSE.unitree_go2 (BSD-3-Clause)` — so a publish never carries a file the
  author did not see;
- **warns** for a *restricted* license (the CC NC / SA / ND family, the GPL
  family), naming the file and the clause, before any network call;
- **refuses** a *blocked* license — one the platform recognises as forbidding
  redistribution: the Max Planck "non-commercial scientific research purposes"
  text that SMPL, AMASS and their siblings carry
  (`LicenseRef-MPG-NonCommercial`), Universal Robots' "Terms for Graphical
  Documentation" (`LicenseRef-UR-Graphical-Documentation`), or a file tagged
  `LicenseRef-AMASS` / `LicenseRef-SMPL` — as a `PublishError` on that file,
  before any network call. The platform refuses the same file at commit; this
  is the fast local copy of that gate, exactly as `uses_custom_js` is;
- **says nothing** about a custom text it cannot classify (`LicenseRef-custom`).
  It is uploaded and shown; the platform cannot know what it permits.

Identification is the SPDX tag on line one when present, else a header match
for the standard texts, else a fingerprint for the recognised non-redistributable
licenses above, else `LicenseRef-custom`. The identifier and the tier table live
in `mjswan/licenses/` with the same contents as the platform's
`@mjswan/licenses` package, kept in step by hand like `name2id_cases.json`; the
platform's copy is authoritative for what a publish is accepted with. The
warning is a warning: there is no acknowledgement flag to pass. A build with no
license files publishes as before, and `publish` says nothing about their
absence: the platform shows nothing for it either, and nobody is nudged.

### 4. `mjswan info` lists the files

Beside the projects, scenes and policies it already prints, with the detected
identifier per file. The standalone SPA does **not** yet render them; that is
the natural follow-up (a GitHub Pages deployment of a Unitree demo has the same
obligation the platform has), and it is out of scope here.

## Considered options

| Option | Why not |
|---|---|
| Declare attributions as manifest keys and store them as JSON on the platform (the first draft) | Transcribes upstream text and puts the notice in a database rather than beside the bytes it covers. Set aside on review in favour of files. |
| The work's `LICENSE` at the build root | The root is the engine's, and the engine must keep shipping its license with every `dist/`. The project directory is where the platform's one project lives anyway. |
| One concatenated `LICENSE` per scene | Cannot be identified per component, so cannot be tiered or labelled; a scene with a robot and a motion clip has two licenses. |
| A `licenses/` subdirectory per scene | The same information one directory deeper, and one more layout rule. The suffix keeps the file where the demo already puts it. |
| Prepend the SPDX tag to copied files too | Alters a text the license asks to be reproduced. Copies stay verbatim; the header match identifies them. |
| A full license-identification library | The sources mjswan is used with either carry a `LICENSE` beside the model or are in the known-assets table; a header matcher plus that table is the whole problem. |
| Detect in `_save_web` | The spec and its directories are gone by then, and the author has no chance to correct what was found. |

## Consequences

- A build from a menagerie or `robot_descriptions` model carries the model's
  `LICENSE` verbatim in its scene directory with no author action; an mjlab
  build carries a generated one from the table; a `mujoco_playground` scene
  built from an XML string with in-memory assets has no directory to search and
  gets a generated file only when its model name is a known one; a build from
  an unknown model carries nothing, silently.
- `manifest.json` and `format` are untouched. `.swn` carries the files already;
  self-hosted `dist/` trees carry them for a future SPA that renders them.
- `examples/demo/main.py::_copy_licenses` is retired: detection copies the
  correct per-model file, and the platform now accepts it.
- `publish` gains a second local gate. Like the custom-JS gate it can only act
  on what the build declares, and that is its job.
- The root `dist/LICENSE` keeps meaning what it means today, and stays out of
  every publish.
- `docs/getting-started/core-concepts.md` (output structure, Licenses),
  `docs/guides/publishing.md` (what is uploaded, the warning and refusal),
  `docs/api/core.md`, the CHANGELOG under Unreleased, and `mjswan/licenses/`
  (the module, with the six SPDX texts as package data) change with it.

## Phased execution plan

1. `mjswan/licenses/`: naming rule, SPDX header matcher and tag reader, tier
   table, license templates, known-assets table; unit tests with fixture texts.
2. `publish.py`: admit the names (project-relative only), advisory content type,
   print / warn / refuse; tests beside `TestPlanPublish`.
3. Detection at `add_scene` / `add_scene_mjlab`: candidate directories, the
   two-parent bound, de-duplication by content; `_save_web` writes the files;
   tests against a temporary directory laid out like a menagerie model and like
   a playground checkout.
4. Author API: `Builder(license=, copyright=)`, `add_project(license=…)`,
   `set_license` / `set_notice`, `add_attribution` / `clear_attributions`;
   tests via `build_manifest`'s output tree.
5. `mjswan info`; retire the demo's `_copy_licenses`; docs and CHANGELOG.

## Acceptance criteria

- [x] A scene loaded from a directory holding a BSD-3-Clause `LICENSE` gets
      `<project>/<scene>/LICENSE.<dirname>` byte-identical to it, and nothing
      else.
- [x] A model whose XML lives outside the menagerie checkout but whose meshes
      resolve into it (playground layout) gets the model's file, not the
      repository's.
- [x] A directory two levels above a repository root contributes no file.
- [x] `add_scene_mjlab` for a Unitree G1 task writes a generated
      `LICENSE.unitree_g1` whose first line is
      `SPDX-License-Identifier: BSD-3-Clause`.
- [x] `Builder(license="Apache-2.0", copyright="2026 Example")` writes
      `<project>/LICENSE` from the template with the tag and the holder.
- [x] `plan_publish` uploads `<project>/LICENSE` and
      `<project>/<scene>/LICENSE.x` with `text/plain`, never the root `LICENSE`.
- [x] `plan_publish` refuses a file tagged `LicenseRef-AMASS` before any
      network call, naming it, and warns for one identified as
      `CC-BY-NC-ND-4.0`.
- [x] `manifest.json` is byte-identical with and without license files;
      `format` is 1; a `.swn` round-trips the files.
