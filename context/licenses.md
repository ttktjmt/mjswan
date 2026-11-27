# License files in the build

`license/`: where `LICENSE` / `NOTICE` files sit in a build, how they are found and identified, and what `publish` does with them. The publish side is also in [cloud.md](cloud.md).

## Why files, and why the build

A build packs a model as `scene.mjz`: the MJCF plus every mesh and texture it names. The `LICENSE` beside the model on disk is not named by the XML, so it is not packed, yet the models mjswan is used with mostly require exactly that file to travel (menagerie's Unitree models are BSD-3-Clause, which asks binary redistributions to reproduce the notice; Apache-2.0 asks for the `NOTICE` contents). Some motion data a tracking task bundles is not permissive at all (LAFAN1 is CC BY-NC-ND 4.0; AMASS-derived clips may not be passed on).

The build is the one place that knows what a model *is* (`spec.modelfiledir`, the directories its meshes resolve to, the mjlab task id), so the declaration is made there, and as **files** in the tree rather than manifest text: the notice sits beside the bytes it covers and the upstream text travels verbatim. The manifest only *names* them (a project's `work`, a scene's `credits`, written by `build/credit.py`), because a static host lists no directory; the texts stay in the files.

## Where they sit

```
dist/
├── LICENSE, NOTICE              the engine's own; never the work's, never published
└── <project-id>/
    ├── LICENSE, NOTICE          the work's license (per project)
    ├── licenses/
    │   └── LICENSE.<component>, NOTICE.<component>   third-party, once per text
    └── <scene-id>/              (older builds wrote LICENSE.<component> here)
```

- **The work's license** is per project: a multi-project document may license each differently. The root is unavailable: `install_spa` copies the engine's Apache-2.0 `LICENSE` there, and that obligation stays with every `dist/`.
- **A third-party component** is one file per component, in the project's `licenses/`, written once per text, so a component every scene carries is one file. A second, different text under one name is numbered `-2`, `-3`, …. Readers still accept the scene directory, where builds before `licenses/` wrote them (a bare scene-level `LICENSE` is labelled with the scene id). `licenses/` reads as a scene directory, which is why no scene may take that id.
- **The naming rule** (`license/path.py`), shared with the platform's validator: basename `^(LICENSE|NOTICE)(\.[A-Za-z0-9_-]{1,64})?$`, at `<project>/` (no suffix) or one directory below. The root is never a license location for `publish`. The platform refuses a file over 64 KiB.

## Finding them: detection at `add_scene`

`add_scene` (and `add_scene_mjlab` and `add_scene_hf`, which call it) records attributions on the scene, while the spec and its directories exist and the author can still correct what was found.

1. **Files beside the model** (`license/attribution.py`: `detect_attributions(spec_asset_directories(spec))`). Candidate directories are `spec.modelfiledir` and those the spec's meshes, textures, hfields and skins resolve to. Each is searched, then its parents up to **two** levels, and the nearest level holding any `LICENSE*`, `COPYING*` or `NOTICE*` wins for that directory. The files are taken verbatim as `LICENSE.<dirname>` / `NOTICE.<dirname>` (at a Hub snapshot's root, the repository's name), de-duplicated by content. Two parents reach a model's directory from its `assets/` (menagerie, `robot_descriptions`, a `mujoco_playground` checkout whose meshes resolve into menagerie); a third reaches a repository root and copies the *repository's* license instead, which is the failure this rule exists to prevent. Paths are made absolute with symlinks kept, since the Hub cache links each file into `blobs/`, away from the license beside the link, and a `~` stays a folder name, as it is to MuJoCo. A spec parsed from a string has no directory and is not searched: resolving its asset paths against the working directory would find whatever happens to be there.
2. **A table of known assets** (`KNOWN_ASSETS`): `{component, spdx, holder, url}`, matched by the tokens of a menagerie directory name, `spec.modelname` or an mjlab task id. When no file was found but the table knows the model, mjswan generates the text from its template with the holder's line. This covers mjlab's asset zoo (Unitree, ANYbotics), whose XMLs ship with no license file beside them.
3. **The author**: `SceneHandle.add_attribution(component, *, license=<spdx or path>, notice=<path or text>, copyright=None)` and `clear_attributions()` for a detection that is wrong. Motion clips and policy weights are only ever declared this way: there is no file on disk to find, and a detector that guessed would be worse than one that stays silent.

The work's own files come from `Builder(license=, copyright=)` as the default for every project, `add_project(name, license=, copyright=)` per project, and `ProjectHandle.set_license()` / `set_notice()`. A generatable SPDX id (the six texts under `license/templates/`: Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, CC-BY-4.0, CC0-1.0) produces the standard text, tagged; a path is copied.

A build from an unknown model with no file beside it carries nothing, silently, and `publish` says nothing about the absence either.

## Identifying them

`license/spdx.py`: the SPDX tag on line one when present (`SPDX-License-Identifier: BSD-3-Clause`), else a match on phrases only one standard text contains, else a fingerprint for the recognised non-redistributable licenses, else `LicenseRef-custom`.

A file mjswan *generates* begins with the tag; a file mjswan *copies* is copied verbatim, with no line added: prepending would alter a text the license asks to be reproduced, and the header match identifies the standard texts anyway. The tag is for what a header cannot settle.

The tier table, kept in step by hand with mjswan Cloud's (the platform's copy decides what a publish is accepted with):

- **`notice`**: everything permissive; printed.
- **`restricted`**: the CC NC / SA / ND family and the GPL family; `publish` warns, naming the file and the clause, before any network call. There is no acknowledgement flag to pass.
- **`blocked`**: licenses that forbid redistribution: the Max Planck "non-commercial scientific research purposes" text SMPL, AMASS and their siblings carry (`LicenseRef-MPG-NonCommercial`), Universal Robots' "Terms for Graphical Documentation" (`LicenseRef-UR-Graphical-Documentation`), recognised from their text, and `LicenseRef-AMASS` / `LicenseRef-SMPL`, from a tag only. `publish` refuses one with a `PublishError` on that file before any network call; the platform refuses the same file at commit, so this is the fast local copy of that gate, as the custom-TS refusal is.
- `LicenseRef-custom`, a text it cannot classify, is uploaded and shown without comment: the platform cannot know what it permits.

`Attribution` is what a scene carries and the build writes; `declare()` is what `publish` and `mjswan info` read back from a built tree. `license/vrm.py` reads a VRM's license as its meta states it ([skins.md](skins.md)). In the browser, the About dialog lists the scene's `credits` and the project's `work`, then the engine's own license ([frontend.md](frontend.md)).

## Rejected

| Option | Why not |
|---|---|
| Attributions as manifest keys, stored as JSON on the platform | Transcribes upstream text and puts the notice in a database rather than beside the bytes it covers. |
| The work's `LICENSE` at the build root | The root is the engine's, and the engine must keep shipping its license with every `dist/`. |
| One concatenated `LICENSE` per scene | Cannot be identified per component, so cannot be tiered or labelled; a scene with a robot and a motion clip has two licenses. |
| Prepend the SPDX tag to copied files | Alters a text the license asks to be reproduced. |
| A full license-identification library | The sources mjswan is used with either carry a `LICENSE` beside the model or are in the known-assets table; a header matcher plus that table is the whole problem. |
| Detect when the scene directory is written | The spec and its directories are gone by then, and the author has no chance to correct what was found. |
