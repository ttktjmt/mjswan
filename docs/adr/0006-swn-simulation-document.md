# The `.swn` simulation document — one manifest, MDP configs, scene-scoped layout

> Status: **Accepted** — supersedes **§1 of
> [ADR 0005](0005-onnx-traced-terms-superseding-the-declarative-dsl.md)**
> ("Representation, not artifacts — extend `config.json` + `policy.json`", which
> states verbatim *"No new `manifest.json`."*) and amends its **§2** (the PRNG is
> reseeded at an MDP switch) and **§4** (one graph-reference resolution base). ADR 0005 §3,
> §5–§9 carry over unchanged. This is a pre-1.0 change to the **build output**:
> what files a build writes, how they are laid out, and what describes them. It
> does not change how a term body is traced or executed.

## Context

ADR 0005 §1 deliberately avoided introducing a new top-level artifact: the
build kept writing `config.json` plus one config JSON per policy
(`<scene-id>/<policy-id>.json`, `builder.py:479-480`), and only the term-body
*representation* inside them changed. That was the right call for a
representation swap. It does not survive contact with three things the output
has since had to carry.

**1. Graph paths collide between policies in one scene.** Every traced graph was
written to a scene-relative `obs/<group>.onnx`, `term/<name>.onnx`,
`command/<name>.onnx`. A group or term name is scene-wide, and two policies in
one scene routinely name their observation group the same thing. The second
write silently won and every config still pointing at the first loaded the wrong
graph, with no error at playback. Commit `599ba9f` fixed this by scoping each
policy's graphs to `<policy-id>/…` and adding a guarded write that fails the
build when two different graphs want one path (`src/mjswan/_graph_io.py`). That
fix is correct and stays, but it treats the *policy* as the owner of an MDP —
which is the second problem.

**2. Checkpoints of one task share an MDP, and the layout cannot say so.** The
dominant workflow is W&B multi-checkpoint comparison. In
`examples/mjlab/defaults/main.py`, one project holds 7 scenes across 11 W&B
runs, and `add_policy_wandb` defaults to `only_latest=False` — every
`model_*.pt` in a run becomes its own policy. All the policies in one scene are
trained against **one** `env_cfg`, so they share one observation set, one
termination set, one command set, one event set — yet the build traces and
writes all of it once per policy, because the policy is the only unit that owns
an MDP. There are 7 distinct MDPs in that example and dozens of copies of them.

Policies are named for the W&B file they came from — `name = pt_path.stem`, so
`model_0`, `model_50`, … (`wandb_io.py:347`) — and mjlab's runs for one task are
a **resume chain**: run *n+1* starts from run *n*'s last checkpoint, so the only
name two runs share is that boundary. `add_policy_wandb` drops the repeat with a
`seen_names` set (`scene.py:729-731`), which is the right answer for a resume
chain and is left alone. Across *scenes*, though, the same names recur in full,
and nothing separates them.

**3. `event` has no owner at all.** `Scene._derive_term_sets` handles
observations, actions, terminations, and commands; events are held on the scene
and are *not* scoped per policy. That is a real modelling gap, not a shortcut:
mjlab keeps events in `ManagerBasedRlEnvCfg` as a sibling of the other four
managers, so an event set belongs to the same unit they do. Because events are
scene-wide today, switching policy in the viewer cannot re-apply domain
randomization, and `EventManager.startup()` constructs a fresh
`ModelFieldDefaults` per pass (`core/event/EventManager.ts:149`), which
snapshots whatever the model currently holds — so a second startup pass after a
first randomization would treat the *randomized* values as the compiled base.

Alongside these, the output has accumulated smaller inconsistencies that are
cheapest to settle in the same change: project directories were the only
identifier not passed through `name2id`; the first project became a literal
`main/` directory rather than an id; per-project `index.html`/`logo.svg` copies
survive from a sub-directory routing scheme the SPA no longer uses (routing is
`?project=`); and the default observation-group key is `"policy"`, a name chosen
before groups were traced to ONNX and before mjlab's own name (`"actor"`) was
understood.

Finally, the intended end state is that a build is a **document** — one
`<name>.swn` file — that can be handed to someone, hosted on GitHub Pages inside
an app shell, or uploaded to mjswan Cloud. A document needs exactly one thing
that describes it.

## Decision

### 1. One `manifest.json` at the document root — supersedes ADR 0005 §1

The build writes a single `manifest.json` at the root of the document. It
replaces `assets/config.json` **and** every per-policy config JSON: policy
metadata is inlined into the manifest rather than living in sibling files.

ADR 0005 §1's reasoning was that a representation change should not move
artifacts. That holds for a representation change. This ADR changes the
artifacts themselves — what is owned by what — and a per-policy config file
cannot express an MDP shared by four policies without either duplicating it or
inventing a cross-file reference. One manifest is the smaller mechanism.

Top-level shape:

```json
{
  "format": 1,
  "version": "0.10.0",
  "projects": [ … ]
}
```

Every key in the manifest is `snake_case`, at every depth. `uses_custom_js` and
the `plugins` reference (ADR 0003's trusted-context custom-JS path) keep their
current meaning and move to the top level unchanged.

### 2. Layout — ownership levels are directories, multi-file kinds are boxes

```
manifest.json
<project-id>/
  <scene-id>/
    scene.mjz
    mdp/
      <mdp-id>/
        obs/<group>.onnx
        term/<name>.onnx
        command/<name>.onnx
        event/<mode>.onnx
    policy/
      <policy-id>.onnx
    assets/
      <motion>.npz
      <splat>.spz
```

Two rules generate the whole layout:

- **Every ownership level gets a directory.** Project and scene are the levels
  at which mjswan guarantees identity, so each is a directory. This is what
  makes `examples/mjlab/defaults` work: its checkpoint policies are named
  `model_0`, `model_50`, … in every scene, so a flat `policy/<id>.onnx` would
  have them overwrite each other — the same failure `599ba9f` just fixed one
  level down. Within one scene they still collide, and §4's rename resolves
  them.
- **Within a scene, only a kind that can hold more than one file gets a box.**
  `scene.mjz` is one file per scene, so it sits directly in the scene directory.
  `mdp/`, `policy/` and `assets/` can each hold many, so each is a directory.

`assets/` keeps the name the build already uses for a project's asset directory
(`builder.py:820`), moved down to the scene that owns the files. Per-project
`index.html` / `logo.svg` copies
(`builder.py:825-834`) are deleted — they served sub-directory routing, and the
SPA selects a project from `?project=` against a build-time base URL
(`App.tsx:42,105`). A URL like `…/mjswan/myosuite/` becomes a 404 rather than
rendering the wrong project, which is the correct outcome.

### 3. `MdpConfig` — the MDP is a named unit, and `event` belongs to it

A new Python class `MdpConfig` holds the five managers together:

```python
mdp = mjswan.MdpConfig(
    observations=…,
    actions=…,
    terminations=…,
    commands=…,
    events=…,
)
scene.add_policy(name="mowqlkd5", policy=…, mdp=mdp)
scene.add_policy(name="sif72y3p", policy=…, mdp=mdp)   # same object → one mdp/ on disk
```

`PolicyConfig` gains an `mdp` field. The existing per-manager keyword arguments
on `add_policy` remain and are sugar: they construct an anonymous `MdpConfig`.
Two policies given the *same* `MdpConfig` object share one `mdp/<mdp-id>/`
directory and one set of traced graphs; the manifest records the reference.

This is the change that makes points 2 and 3 of the Context one change rather
than two. `events` becomes the fifth field of `MdpConfig` and
`Scene._derive_term_sets` (`scene.py:343`) grows from four fields to five,
matching mjlab's `ManagerBasedRlEnvCfg`. Events stop being scene-wide, so
switching policy switches the event set with everything else.

Identity of an `MdpConfig` is **by object**, not by content. Content-addressed
pooling of graphs was considered and rejected (see *Considered options*).

Its id follows from that. `MdpConfig` takes an optional `name`; when given, the id
is `name2id(name)`. When absent, the id says whose the MDP is. One the sugar path
constructs exists for the single policy it was built for and takes **that policy's
id**, so `mdp/locomotion/` sits beside `policy/locomotion.onnx` and a scene's MDP
directories read as its policies do. One handed to several policies by hand — what
`add_policy_wandb` does for a run's checkpoints — belongs to no single policy and is
`mdp_<n>`, numbered from zero **per scene** in the order the MDPs are first used. So
a scene whose policies all share one config writes `mdp/mdp_0/`, and
`examples/mjlab/defaults` can name its MDPs after the task to get
`mdp/velocity_rough/` instead. Numbering per scene, not per document, keeps a
scene's ids stable when another scene is added before it.

> **Amended 2026-09-02.** The fallback was `mdp_<n>` for every unnamed config, the
> sugar path's included. A scene with several single-policy MDPs then wrote `mdp_0`,
> `mdp_1`, `mdp_2` with the policy names sitting right beside them in `policy/`.

### 4. Identifiers — sanitized names, scoped uniqueness, rename on collision

Every id in the document is `name2id(name)` (`utils.py:358`): lowercased, every
run of non-`[a-z0-9]` collapsed to `_`. This was already true of scenes,
policies, splats and motions; projects were the exception and are brought in
line. The `id=` parameter of `add_project` is removed — one object, one
identifier, derived one way.

Uniqueness is scoped to the level that owns the directory:

| id | unique within |
|---|---|
| project | the document |
| scene | its project |
| mdp, policy, asset | its scene |

A collision within a scope is **not** an error. The build renames the second
occurrence to `<id>_1`, the third to `<id>_2`, and emits a warning naming both
the original name and the assigned id. Two scenes called "Flat Terrain" is a
reasonable thing to write and should not fail a build.

Immutable generated ids (UUIDs) were considered and rejected: a URL is a sharing
surface and `?project=g1_locomotion&scene=flat_terrain` has to be readable by a
person. mjswan Cloud simulations are immutable once published — changing one
means re-publishing — so a human-readable id that could in principle be reused
across builds does not break a shared link.

The URL parameters `?project=`, `?scene=` and `?policy=` resolve against these
ids. `pickByName` (`App.tsx:50`) matches the sanitized form only; the raw-name
fallback is removed.

That last part exposes a divergence that has been latent. The frontend's
`sanitizeName` is **not** equivalent to Python's `name2id`:

```
name2id     re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
sanitizeName  name.toLowerCase().replace(/ /g,'_').replace(/-/g,'_')
```

They agree on spaces and hyphens and disagree on everything else:

| name | `name2id` (the directory) | `sanitizeName` (the URL match) |
|---|---|---|
| `Newton's Cradle` | `newton_s_cradle` | `newton's_cradle` |
| `G1 (with hands)` | `g1_with_hands` | `g1_(with_hands)` |
| `Café` | `caf` | `café` |

Today `pickByName` falls back to the raw name, so the mismatch is invisible.
Once the fallback goes, a link to any such scene resolves to the default instead.
`sanitizeName` is rewritten to mirror `name2id` exactly, and a test pins the pair
against a shared table of cases — the comment on `sanitizeName` already claims it
mirrors the Python helper, which is the bug.

### 5. Observation group keys, and what `in_keys` actually is

`in_keys` is **the ONNX input slot assignment table**: `in_keys[i]` names the
tensor that fills the session's *i*-th input. `OnnxModule.runInference` pairs
them positionally (`core/policy/OnnxModule.ts:67-73`); the model's own input
names are read from the session at run time and never appear in any config.

It is not a list of observation groups. In `examples/demo`, the go2 policies
declare four slots — `['policy', 'is_init', 'adapt_hx', 'command_']` — of which
two are observation groups and two (`is_init`, the recurrent carry `adapt_hx`)
are synthesized by the runtime (`OnnxModule.ts:49-52`, `runtime.ts:1396-1398`).
The observation dict's key order cannot express that, because two of the four
slots are not in it.

Consequences:

- **The default observation group key becomes `"actor"`**, mjlab's own name for
  the group its actor network reads (`mjlab/tasks/velocity/velocity_env_cfg.py`
  keys `observations` by `"actor"`/`"critic"`; `mjlab/rl/config.py:92` maps
  network → group). `DEFAULT_OBS_GROUP_KEY` (`adapters/mjlab_adapter.py:174`)
  and the frontend fallback (`OnnxModule.ts:30`) change **together** — either
  alone breaks every policy that declares no `in_keys`.
- **`in_keys` is dropped for single-input policies.** For a session with one
  input, `inferInputKeys` returns the configured keys without consulting the
  model at all (`OnnxModule.ts:107-109`), so the label is free and a declaration
  adds nothing. Five of the eight `examples/demo` assets are single-input and
  lose the field.
- **`in_keys` is retained for multi-input policies**, where it is the only
  record of where the runtime-synthesized tensors sit relative to the
  observation groups. Those three assets keep it, with their group renamed
  `policy` → `actor` for consistency.
- The fused graph file is named for its group, so the common case becomes
  `mdp/<mdp-id>/obs/actor.onnx`. A group that cannot be fused
  (`_group_is_fusable`, or a `ConstantGroup` bail at trace time) still writes
  one file per term under the same directory and still occupies one slot.
- The `name === 'obs' && configuredInKeys.includes('policy')` special case
  (`OnnxModule.ts:121-123`) is deleted. It was a fossil of the `"policy"`
  default.
- The `PolicyConfig.observations` docstring (`policy.py:50`) currently says
  the keys are "**ONNX input names**". That is wrong — `g1/locomotion.onnx`'s
  real input is `observation` while its config declares `["policy"]`, and the
  go2 models' real inputs are `l_kwargs_policy_` / `arg1` — and it is corrected
  as part of this change.
- A config whose `in_keys` length disagrees with the session's input count is a
  **build-time error**. Today it fails at playback with `Missing ONNX input for
  key`.

**The slot tables live in the manifest, declared in Python.** `in_keys` and its
sibling `out_keys` (mapped the same way onto `session.outputNames`) become
arguments on `add_policy` and are written into the manifest's policy entry:

```python
scene.add_policy(
    name="Facet",
    policy=onnx.load("assets/unitree_go2/facet.onnx"),
    observations={"actor": …, "command_": …},
    in_keys=["command", "actor", "is_init", "adapt_hx"],
    out_keys=[…],
)
```

They stop being read from a `config_path` sidecar, and the top-level
`in_keys` / `out_keys` promotion into `onnx.meta` (`builder.py:635-636`) is
deleted with them. `config_path` stays an **authoring** input for the
checkpoint's own defaults — `policy_joint_names`, `default_joint_pos`,
`actions`, `clip_actions` — merged at build time; it stops being a place the
*output* reads from, because §1 leaves exactly one descriptor in the output.

This is affordable precisely because every script under `examples/` is
restructured for this architecture anyway. The one cost is visible: facet's
`out_keys` is sixteen entries, most of them placeholders for outputs mjswan
never reads, and in a script it is verbose where a sidecar hid it. Visible is
the better failure mode — a positional table that nobody can see is the reason
`g1/locomotion.json` has declared the wrong name for its input all along
without anyone noticing.

### 6. Graph references resolve against the scene directory — amends ADR 0005 §4

ADR 0005 §4 fixed *what* is fused; it left the reference form to §1's
`"onnx": "obs/base_ang_vel.onnx"`. With graphs now under
`mdp/<mdp-id>/`, every graph reference in the manifest is a path **relative to
its scene directory**, and that is the only resolution base:

```json
{ "onnx": "mdp/default/obs/actor.onnx" }
```

One base, applied everywhere, replaces the current mixture of scene-relative
graph refs and root-relative asset paths. `src/mjswan/_graph_io.py` keeps
sole ownership of both the path convention and the guarded write; its collision
guard stays as a backstop even though the layout now makes collisions
structurally impossible within a scene.

### 7. `format` and `version` are independent, and neither is `min_viewer`

- **`version`** is the mjswan version that wrote the document. The release
  workflow sets `src/mjswan/__init__.py` and `template/package.json` from one
  input (`.github/workflows/publish-npm.yml:37-39`), so the Python version and
  the npm engine version are the same string by construction, and
  `https://cdn.jsdelivr.net/npm/mjswan@<version>/dist/mjswan.js` always resolves
  for a released version. A host uses `version` to **select the engine**.
- **`format`** is a manually incremented integer describing the document's
  structure. A reader refuses a document whose `format` exceeds the highest it
  knows. **An absent `format` means the pre-0006 layout** (root
  `assets/config.json`, per-policy JSONs); this ADR's layout is `format: 1`.
  The phases below land in one release, so no document is ever published at an
  intermediate value. `format: 2` (#127) added raw-`mjData` input slots
  (`{"sim": ...}` with `rows`), which a format-1 engine accepts and then cannot
  serve; the layout is otherwise this ADR's. It is the guard for the cases where engine selection is not available:
  a document opened by whatever engine is already present, and third-party
  tooling reading the tree.

The two are not redundant because they are read by different parties for
different decisions — one by the host choosing an engine, one by the engine
protecting itself. A document therefore carries both.

No `min_viewer` key and no `FEATURE_SINCE` table. A per-feature minimum-viewer
computation was designed and dropped: it only ever helps the "old engine, new
document" pair, which engine selection eliminates for the hosted cases, and it
costs a hand-maintained table that must stay correct forever. Adding it later is
backward compatible (an old engine ignores an unknown key) but cannot
retroactively help engines already shipped.

For hosts that select an engine, a **version override table** — empty by
default — maps a document's `version` to a different engine version:

```json
{ "0.9.3": "0.9.7" }
```

Resolution is `overrides[version] ?? version`. Nothing to maintain in the normal
case; one row when a published engine version turns out to be broken, pointing
at an already-published newer one. No backporting, no new release line, no
rewriting of stored documents. The same row form handles a document stamped with
an unpublished development version.

mjswan Cloud's intent is to **migrate stored documents forward** so everything
runs the newest engine, rather than pinning each to the engine that wrote it; the
override table is then the fallback for the documents migration cannot reach.
See *Migrating stored documents* for which those are. `format` is required under
either policy — a migrator needs it to know what it is reading, and a pinning
host needs it as the engine's own guard.

### 8. The `.swn` container, and the two output modes

A simulation document is a ZIP container with the extension **`.swn`**, holding
the tree of §2 at its root. ZIP is chosen for the same reason `.docx`, `.pptx`
and partitioned Parquet directories choose it: a directory of files is the
authoring model, and the container is a reversible packaging step over it, so
every tool that already understands the directory keeps working.

Two output modes:

- **Document** — `<name>.swn`. The deliverable.
- **App** — a directory containing the engine plus the **expanded tree**, not
  the zip. This is what runs from a local `mjswan serve` and what uploads to
  GitHub Pages unchanged. Handed a document, `mjswan serve` builds this mode
  transiently: it expands the tree into a temporary directory, lays the packaged
  engine over it, and drops it at exit. A custom-JS build is refused there, its
  plugin module being part of the engine rather than of the document. Embedding the zip would force the page to unpack it in
  the browser before the first byte of the scene could load, for no benefit
  where the files can simply be served.

mjswan Cloud continues to receive **individual files**, not a `.swn`. The
publish protocol (`upload-session` → per-file presigned R2 PUT → `commit`,
`MAX_FILES=64`, `MAX_FILE_BYTES=50MB`, `MAX_TOTAL_BYTES=200MB` —
`src/mjswan/publish.py`) already uploads a file tree with per-file content types
and per-file limits. Uploading one archive instead would require a server-side
unpack, lose per-file `Content-Type` for R2 delivery, put every document under a
single 50MB PUT, and forfeit incremental publishing. When the CLI is handed a
`.swn` it unpacks it locally and publishes the tree — zero server change.

### 9. Domain randomization across an MDP switch — amends ADR 0005 §2

Because §3 makes the event set switch with the policy, switching must re-run
`mode="startup"` events, and re-running them must not compound.
`ModelFieldDefaults` (`core/event/modelFieldDr.ts:52`) snapshots each model
field on first touch, and `EventManager` constructs one per startup pass
(`EventManager.ts:149`). It is **hoisted to scene lifetime**, so the compiled
values remain the base across any number of switches.

An MDP switch is, in order:

1. restore every snapshotted model field to its compiled value;
2. **reseed the orchestrator PRNG to the session seed**;
3. apply the incoming MDP's `startup` events;
4. re-derive dependent quantities — `mj_setConst`, joint/actuator bounds.

The reseed is the amendment to ADR 0005 §2, and it is not a new idea — it is the
rule the runtime already applies to the event of the same class. `loadEnvironment`
does exactly this today:

```ts
// runtime.ts:393 — "Reseed so two loads of the same scene draw the same randomness."
this.termRng = new SeededRng(this.termSeed);
```

An MDP switch is a scene load in every way that matters to randomness, so it gets
the same treatment. `termSeed` is the session's own seed (`runtime.ts:664`), held
`readonly`, so the consequences are:

- A scene's randomization is a function of **(session seed, MDP)** alone — not of
  how long playback ran before the switch, and not of the switch history.
- Switching away from an MDP and back reproduces its first draw exactly, which is
  what makes a switch testable at all.
- Two MDPs with the same domain-randomization terms land in the **same** world,
  which is the right default for the workflow this ADR exists to serve: comparing
  policies means comparing them under one set of conditions, not two.

§2's substance — orchestrator-owned PRNG, `rand` threaded as an explicit graph
input, no ONNX random ops — is unchanged.

### 10. What ADR 0005 keeps

§3 (stateful terms promoted to explicit `(prev_state) → (next_state, …)`),
§5 (port trigger semantics, N=1), §6 (dtype at the WASM↔ONNX boundary),
§7 (internals mirror mjlab, external API preserved), §8 (the authoritative step
loop) and §9 (traceability and non-traceable terms) are untouched. §4's fusion
rules are untouched; only the reference base moves (§6 above). The
five-manager restriction and the effect-free / Cloud-safe posture inherited from
ADRs 0003/0004 are not relitigated.

## Manifest schema

The envelope, not the term internals. Every observation / termination / command /
event entry keeps the shape ADR 0005 §1 gave it — `{name, onnx, size,
input_slots, noise, scale, clip, history_length}` for a term, `{fused,
input_slots, native_inputs, layout, size, sensors?}` for a fused observation
group (`_onnx_build.py:305-312`), a list of per-term entries where a group could
not be fused. None of that changes. What this section fixes is what surrounds
them, because that is what a per-policy JSON used to define implicitly.

### Rules that hold everywhere

1. **Every key is `snake_case`, at every depth.** The current config carries
   `splatSection`, `terrainData` and `controlDt`; all three are renamed.
2. **A path resolves against the directory of the level that declares it.** A
   path under a scene entry — `scene`, every graph ref, every asset — resolves
   against `<project-id>/<scene-id>/`. Only the top-level `plugins` resolves
   against the document root, because that is the level it is declared at. This
   is §6's single resolution base, stated for all paths rather than for graphs
   alone.
3. **A default is a `"default": true` flag on the entry**, matching how policies
   already mark one. At most one sibling may set it — two is a build error, not a
   silent pick — and when none does, the first in document order is the default.
   Projects and policies both work this way.
4. **A key is omitted when it carries the default**, so `in_keys` is absent when
   it is `["actor"]` and `out_keys` when it is `["action"]`. The reader supplies
   the default; the writer never emits noise.

### Shape

```json
{
  "format": 1,
  "version": "0.10.0",
  "uses_custom_js": false,
  "plugins": "assets/plugins.js",
  "projects": [
    {
      "id": "mjlab_tasks",
      "name": "mjlab Tasks",
      "default": true,
      "scenes": [
        {
          "id": "mjlab_velocity_rough_unitree_g1",
          "name": "Mjlab-Velocity-Rough-Unitree-G1",
          "scene": "scene.mjz",
          "control_dt": 0.02,
          "camera": { "lookat": [0, 0, 0], "distance": 4.0, "…": "ViewerConfig" },
          "terrain_data": { "…": "TerrainData" },
          "splat_section": true,
          "splats": [
            { "id": "street", "name": "Street", "path": "assets/street.spz",
              "control": true, "transform": { "…": "SplatTransform" } }
          ],

          "mdps": [
            {
              "id": "mdp_0",
              "observations": {
                "actor": {
                  "fused": "mdp/mdp_0/obs/actor.onnx",
                  "input_slots": [], "native_inputs": [], "layout": [], "size": 48
                }
              },
              "actions":      { "joint_pos": { "kind": "…", "target_ids": [], "scale": 0.5 } },
              "terminations": { "fused": "mdp/mdp_0/term/terminations.onnx", "…": "" },
              "commands":     { "velocity": { "onnx": "mdp/mdp_0/command/velocity.onnx", "…": "" } },
              "events":       [ { "name": "dr_friction", "mode": "startup",
                                  "onnx": "mdp/mdp_0/event/startup.onnx", "…": "" } ]
            }
          ],

          "policies": [
            {
              "id": "model_2000",
              "name": "model_2000",
              "default": true,
              "mdp": "mdp_0",
              "onnx": "policy/model_2000.onnx",
              "in_keys": ["command", "actor", "is_init", "adapt_hx"],
              "out_keys": ["…"],
              "policy_joint_names": [], "policy_num_actions": 12,
              "default_joint_pos": {}, "encoder_bias": null,
              "clip_actions": 100.0, "initial_qpos": [], "initial_qvel": [],
              "extras": {}, "source": "…",
              "motions": [
                { "name": "walk", "default": true, "path": "assets/walk.npz" }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

### What moved, and why

- **`policies[].config`** — the relative path to a per-policy JSON — is gone.
  Everything it pointed at is inlined on the policy entry. This is the change
  that lets four checkpoints share one `mdp` reference instead of four copies of
  one file.
- **`scenes[].events`** moves from the scene entry into `mdps[].events`. It is
  the same list; only its owner changes (§3).
- **`actions`** lives in the MDP, not on the policy. `examples/demo` shows this
  holds under strain: the three go2 policies share `go2_actions` while Facet's
  observations differ, so Facet gets its own MDP that names the same actions.
  In Python that is one object referenced twice; in the manifest it is one small
  declarative block written twice, which is the correct trade against making
  actions a fourth ownership level.
- **`id` joins `name` on every entry.** `name` is what a person reads; `id` is
  the directory, the URL parameter, and the reference target. Today the two are
  conflated and the frontend re-derives one from the other at read time
  (`sanitizeName`), which is what §4's divergence is about.
- **`mdp` is always written**, even when a scene has one MDP. A reader that never
  needs a fallback cannot get the fallback wrong.

### Reading it

`src/mjswan/template/src/manifest/index.ts` holds both halves of the contract —
the `Config*` interfaces that mirror what Python writes, and the `Catalog` /
`SceneEntry` / `PolicyEntry` interfaces the app actually holds. Only the first
half changes shape. `PolicyEntry.build()` stops fetching a policy JSON and reads
its entry from the manifest already in memory, which removes one round trip per
policy switch.

## Considered options

**Keep `config.json` + a per-policy config JSON, scope graph paths per policy.** This is
`599ba9f` and it is already shipped. It fixes the collision but keeps the policy
as the MDP's owner, so every checkpoint of a task still traces and writes its own
copy, and `event` still has no owner. Correct as a bug fix, insufficient as a
model.

**Content-address the graphs and pool them.** Write each traced graph to
`graphs/<sha256>.onnx` and let every config reference it. This deduplicates the
checkpoints without introducing any new unit — and was **rejected**. It
makes every path in the document opaque, so nothing can be found or diffed by
hand; it makes the tree unreadable to a person opening the zip; and it solves
only the duplication, leaving `event` unowned. Deduplication is a consequence
worth having, not a thing to organize the document around.

**Generated immutable ids (UUIDs) for projects, scenes and policies.**
Guarantees uniqueness with no rename logic and makes links stable under
renaming. **Rejected**: URLs are read by people, and Cloud documents are
immutable once published, so the stability that UUIDs buy is stability the
system already has.

**A per-file `min_viewer` computed from the features used.** Detects "an old
engine will silently ignore something in this document". **Rejected** — see §7.

**Upload a single `.swn` to Cloud.** Simplest possible publish call.
**Rejected** — see §8.

## Consequences

- `examples/mjlab/defaults` traces 7 MDPs — one per scene — instead of one per
  checkpoint. Build time and document size fall by the checkpoint multiplicity,
  which is the dominant term for a W&B comparison build.
- Deduplication and renaming stay separate concerns, which is worth stating
  because they look alike. `add_policy_wandb`'s `seen_names` drops a checkpoint
  that is *the same checkpoint* seen twice at a resume boundary. §4's rename
  handles two *different* objects that happen to sanitize to one id. Neither
  should grow into the other.
- `event` becomes switchable per policy, which is the feature this reorganization
  exists to enable.
- A document is one file with one descriptor, readable by unzipping it.
- **Every consumer of `config.json` breaks in one release.** `mjswan/manifest`,
  `publish.py` (`_find_config` looks in `assets/config.json` then
  `config.json`), `_cli.py:426`, the docs, and the Cloud reader all move to
  `manifest.json` at the root. This is a pre-1.0 break taken deliberately and all
  at once rather than behind a compatibility shim.
- Existing published Cloud documents are `format`-less and root-`config.json`
  based. They are migratable — see *Migrating stored documents* — because nothing
  in this ADR reaches inside a traced graph.
- The `?project=` value changes for any project whose name was not already
  lowercase-alphanumeric. Existing shared links to such projects fall back to the
  default project rather than erroring.
- `examples/demo/main.py` simplifies: the ANYmal policy's hand-written
  `observations={"obs": …}` workaround (`main.py:419`) becomes a bare
  `ObservationGroupCfg`, and five asset JSONs lose their `in_keys` line.

## Phased execution plan

Phases are commit boundaries within one release, not separate releases.

0. **Stamp `format` and `version`** into the existing `config.json`. No layout
   change. Establishes the gate before anything depends on it.
1. **Identifier discipline.** `name2id` for projects, remove `add_project(id=)`,
   remove the `main/` special case, collision rename with warning, URL params
   resolved against the sanitized name only. Delete the per-project
   `index.html`/`logo.svg` copies.
2. **`MdpConfig`** in Python, with the build still emitting the current layout
   byte-for-byte. Pure refactor; the output is unchanged, so the diff is
   reviewable against a golden build.
3. **New layout + `manifest.json`.** The mechanical move, including the
   per-scene `assets/` directory and the single graph-reference base.
4. **Events move into `MdpConfig`** — the riskiest phase, because it changes
   runtime behaviour: `ModelFieldDefaults` hoisting, the four-step switch
   sequence, and the PRNG rule at a switch.
5. **Trace cache**, keyed on `MdpConfig` identity, so shared MDPs trace once.
6. **`.swn` packaging** plus the app mode that carries the expanded tree, and
   the publish path that unpacks a `.swn` before uploading.
7. **Observation key change**: `actor` default in Python and TypeScript
   together, `in_keys` dropped for single-input policies, the slot tables moved
   from `config_path` to `add_policy` arguments, the `obs`→`policy` special case
   deleted, the build-time slot-count check added. Every script under
   `examples/` is restructured in this phase.
8. **Downstream docs and the published skill.** `skills/mjlab-to-mjswan/` is a
   plugin in the `ttktjmt` marketplace that teaches this API; `SKILL.md:131`
   lists the four term-set arguments `add_policy` accepts, which is correct today
   and wrong the moment `MdpConfig` lands. It ships with the same release, along
   with `docs/`, `CONTEXT.md` and `CHANGELOG.md`.

## Migrating stored documents

The intent for mjswan Cloud is that every stored simulation is migrated forward
so it runs on the newest engine, rather than each being pinned to the engine that
wrote it. This section is the reference for writing that migration script.

### What R2 holds, and what a migrator can do to it

Cloud stores exactly what `plan_publish` uploads — `.json`, `.mjz`, `.onnx`,
`.npz`, `.ply`, `.spz` (`publish.py:41-43`).

| Kind | What it is | Migratable |
|---|---|---|
| `manifest.json`, policy JSONs | mjswan-authored structured data | Yes — rewrite freely |
| `.npz` motions | plain arrays | Yes |
| `.ply` / `.spz` splats | geometry | Yes; no reason to |
| `.onnx` policy networks | third-party trained weights | Untouched — ORT reads them by opset, not by any mjswan contract |
| `.mjz` scenes | MuJoCo's format | In principle, but it needs MuJoCo in the migrator; treat as out of scope |
| `.onnx` traced MDP graphs | mjswan compiler output | **Conditionally** — see below |

### The traced graphs: torch's body, mjswan's interface

The graph *body* is `torch.onnx.export`'s output and does not depend on mjswan.
The graph's **interface does**, entirely, and the interface is what a migration
would have to change:

- Input names are mjswan's, not torch's: `_slot_input_name` (`tracer.py:530`)
  produces `sensor__<name>`, `command__<name>`, `<namespace>__<name>`.
- mjswan appends its own inputs: `rand` for event terms (`tracer.py:1072`),
  `resample_mask` + `rand` + the previous-state tensors for command terms
  (`tracer.py:1373`).
- The batch-axis convention is mjswan's: `dynamic_axes={n: {0: "batch"}}`
  (`tracer.py:379`).
- The manifest's `input_slots` (`slots_json`, `tracer.py:588`) is the contract
  telling the runtime what to feed each input, and a fused observation graph also
  carries `layout` — the per-term widths in concat order (`_onnx_build.py:309-310`).
- `slots_json` re-reads the exported graph's real inputs and **drops slots the
  exporter folded into constants** (`tracer.py:600-604`). A graph and its slot
  list are therefore a matched pair produced by one specific torch version; you
  cannot derive either from the other.

So the rule for a migrator is not "graphs can't be touched", it is:

> **A traced graph migrates as long as the change is a rewrapping of computation
> the graph already contains. It does not migrate when the new engine needs
> computation the graph does not contain.**

Rewrapping — mechanical, safe to script with `onnx`:

- renaming or reordering graph inputs and outputs (rewrite `graph.input[i].name`
  and the node references, and the manifest's `input_slots` alongside);
- inserting a `Cast` at a boundary whose dtype convention changed (§6);
- slicing a fused observation output back into per-term outputs, using `layout`;
- composing several graphs into one.

Not migratable — needs the Python term body, which is deliberately never uploaded
(ADR 0005: no term's Python source ships to the browser):

- a term body that must now read an input it was never traced with;
- a change to what a state tensor *means* (§3);
- a change to how randomness is threaded (§2), beyond renaming the `rand` input;
- recovering independent per-term graphs from a fused one (slicing the output is
  not the same thing).

Everything ADR 0006 itself changes — paths, the manifest shape, the group key
rename `policy` → `actor`, dropping `in_keys` — is metadata and file naming.
None of it touches a graph's interior. **Documents already in R2 migrate
forward under this ADR.**

### Guidance for the migration script

1. **Migrate lazily on first read, and cache the result.** A bulk rewrite over
   every stored simulation has a blast radius of "all of them" and fails quietly;
   lazily, a bad migration is one broken simulation, visible immediately, and the
   fix is to drop the cache.
2. **Never overwrite the original upload.** Write the migrated tree beside it and
   serve that. This keeps "a published simulation is immutable" literally true —
   what is immutable is the bytes the author uploaded — and makes a buggy
   migrator revertible by deletion rather than by re-upload.
3. **Record provenance in the migrated manifest.** Keep the original `format` and
   `version` (e.g. under `built_with`) so the migration is idempotent, re-runnable
   after a migrator fix, and auditable.
4. **Migrate one `format` step at a time.** `1 → 2 → 3` composed from small
   functions, not `1 → 3` written directly. Each step is testable against a
   stored fixture, and a new format costs one function.
5. **Leave `.mjz` and the policy `.onnx` alone.** Neither is mjswan's format;
   pulling MuJoCo into the migration path buys nothing.
6. **Refuse rather than guess.** A document the migrator cannot bring to the
   current `format` is served by a pinned engine through the §7 override table,
   or reported to its author to re-publish from source. Both are better than a
   half-migrated document that loads and behaves subtly wrong.

Migration and `format` are complements, not alternatives: the migrator needs
`format` to know what it is reading, and the engine needs it as its own guard for
the documents migration cannot reach.

## Acceptance criteria

- [x] `examples/mjlab/defaults` builds with one `mdp/` directory per scene, not
      one per policy.
- [x] `examples/demo` builds and plays identically to the current build for all
      four projects, including the three multi-input go2 policies.
- [x] Two scenes given the same name in one project produce `<id>` and `<id>_1`
      with a warning naming both.
- [x] Switching policy between two MDPs with different model-field
      randomization, and switching back, restores the compiled model values
      exactly — no compounding.
- [ ] Switching to an MDP after 10 seconds and after 10 minutes of playback
      applies the same model-field randomization, and switching A → B → A
      reproduces A's first draw.
- [x] `sanitizeName` and `name2id` agree on a shared table of cases, apostrophes,
      parentheses and accents included, and a link to `?scene=newton_s_cradle`
      opens that scene rather than the default.
- [x] `skills/mjlab-to-mjswan/SKILL.md` describes the API the release ships.
- [x] No `in_keys` or `out_keys` is read from a `config_path` sidecar; the
      multi-input go2 policies declare theirs in `examples/demo/main.py` and
      they appear in `manifest.json`.
- [x] A document whose `format` exceeds the engine's maximum is refused with an
      error naming both values and the document's `version`.
- [x] `mjswan publish` accepts either a built directory or a `.swn` and uploads
      the same file set for both.
- [x] Two sibling entries both marked `"default": true` fail the build; none
      marked resolves to the first in document order.
- [x] No `config.json` and no per-policy config JSON remains in a build output,
      and no source file outside the migration reads them.

### Verification status

Run on this branch, 2026-09-02. `examples/mjlab/defaults` (7 tasks, 76 W&B checkpoints)
and the local-asset half of `examples/demo` (`mjswan Demo` + `MyoSuite`, 11 scenes) build,
and every scene of both loads and plays in headless Chromium: `window.__mjswanReady`, no
console error, a non-blank frame that moves. The three multi-input go2 policies play as
the deployed pre-ADR-0006 build does, their collapse included, so the slot tables carried
over faithfully. `examples/demo/gentle_humanoid` builds and tracks its clip only once its
checkpoint's `out_keys` moved from the sidecar to `add_policy`: without them the runtime
drove the actuators from output 0 and the robot fell, which is why a multi-output network
without `out_keys` now warns at `add_policy`.

`examples/demo`'s four projects then built in full, 120 scenes, and a two-scene app named
`Simple Pendulum` / `Newton's Cradle` confirmed the URL half of the id rule: no query opens
the first scene, `?scene=newton_s_cradle` and `?scene=Newton's%20Cradle` both open the
second. No example ships a scene by that name — it is the docs' worked example of
`name2id` — so the case needed a build of its own to exercise.

Still open: the timing half of the randomization criterion rests on the reseed at
`runtime.ts:862` and the `rng` / `modelFieldDr` unit suites, not on a switch driven in a
browser.
