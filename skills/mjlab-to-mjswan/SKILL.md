---
name: mjlab-to-mjswan
description: Port an mjlab task to mjswan: turn a local or remote GitHub repo that registers mjlab tasks into a browser app. Use when the user wants to run, visualize, or deploy an mjlab task (theirs or a third party's) in the browser with mjswan.
---

Port one mjlab task from a target repo into a browser app built by [mjswan](https://github.com/ttktjmt/mjswan). The target is any repo that registers mjlab tasks, given as a local path or a GitHub URL.

**Scope.** Inside the target repo everything you write stays on the Python side: no TypeScript (`ts_src`), no `ViewerConfig`. mjswan itself is edited in exactly one case: a missing capability that is generic rather than task-specific, which step 8 turns into a pull request.

## Ground rules

- mjswan reimplements none of mjlab's term functions: a task's real observation / termination / event / command functions are traced to ONNX at build time and run in the browser beside the policy. A fix therefore means *making mjlab's own function traceable*, never rewriting its math.
- When something fails, read the exception in full, then the module it came from in the installed package: `python -c "import mjswan, pathlib; print(pathlib.Path(mjswan.__file__).parent)"`. That source is the only current truth.
- Two fix attempts per term, then record it as skipped and move on. A partial port that builds and passes parity is the deliverable; all-or-nothing is not.

## 1. Acquire the target

A GitHub URL → `git clone <url> ./<repo-name>`, then work inside it. A local path → use it as is.

Then make mjswan importable from the **same interpreter** that can import the target's task registrations:

- Repo already has an environment → add to it: `uv pip install mjswan torch onnxruntime`.
- Fresh clone with nothing → `uv venv && uv pip install -e . && uv pip install mjswan torch onnxruntime`.
- No `uv` available → install into the interpreter that already runs the target, against its own `sys.prefix`.

mjswan pins `mujoco` **exactly** and bounds `requires-python`. If the install fails on either, **stop and report the resolver's output verbatim**: resolving it is the user's call, not yours.

## 2. Find the task ids

Registration happens as an import side effect, and every repo does it differently: `mjlab.tasks`, a `<pkg>/tasks/__init__.py`, a `src/tasks/` on `sys.path`, or a `bootstrap_*()` function that must be *called*, sometimes only after an env var or a data file is in place.

Look at the repo's own README run instructions first, then grep for modules touching `mjlab.tasks.registry` or calling `register`. Confirm each candidate by diffing the registry around the import:

```sh
python - <<'PY'
from mjlab.tasks.registry import list_tasks
before = set(list_tasks())
import your_repo.tasks             # or: from ... import bootstrap_fn; bootstrap_fn()
after = set(list_tasks())
print(f"{len(before)} already registered, {len(after - before)} new")
print(sorted(after - before))
PY
```

An empty diff means the candidate registers nothing of its own: wrong candidate, or its registration needs a call and arguments you have not supplied yet.

Show the discovered task ids to the user and let them pick one. Ask **here**, not later, for anything the registration itself needs: env vars, clip or dataset paths, gated-dataset access, credentials.

## 3. Pre-flight, seconds before any build

`load_env_cfg` only reads config, so use it to find in seconds what a full build would take minutes to reach:

```sh
python - <<'PY'
from mjlab.tasks.registry import load_env_cfg
from mjswan.mjlab import adapt_actions
import your_repo.tasks
cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1", play=True)
for name, term in (adapt_actions(cfg.actions) or {}).items():
    print("action", name, type(term).__name__, term.unsupported_reason)
PY
```

An action term with a non-`None` `unsupported_reason`, or one `adapt_actions` warns it is **skipping**, makes the task unportable as configured. The action layer is permanently native in mjswan, so there is no author-side escape hatch. Report the reason verbatim, then go to step 8, which decides whether mjswan should learn it. Never hardcode which action classes those are; the `unsupported_reason` field is the source of truth.

## 4. Get the policy into ONNX

- **W&B run path** → nothing to do here. `add_policy_wandb(run_path)` converts every `model_*.pt` itself and passes the metadata below on its own.
- **Local `model_*.pt`** → copy `export_policy.py` (next to this file) into `<repo>/mjswan_app/` and run it. It writes one `<stem>.onnx` per checkpoint plus `policy_meta.json`.
- **Pre-exported `.onnx`** → usable directly, but you have no metadata; see below.

`policy_meta.json` carries `policy_joint_names`, `default_joint_pos` and `encoder_bias`. None of the three can be recovered from an ONNX file, and `add_policy` needs them to resolve action scales and the browser's external PD gains. **A port that omits them builds fine and moves wrongly**, so always pass them, and say so explicitly if a pre-exported `.onnx` left you without them.

## 5. Generate the port

In the target repo:

```
mjswan_app/
  main.py           builder wiring; the only entry point
  terms.py          register_* replacements (only when step 6 needs one)
  export_policy.py  the copied converter        (local .pt only)
  model_*.onnx      converted checkpoints       (local .pt only)
  policy_meta.json  checkpoint order + metadata (local .pt only)
  README.md         source URL, prerequisites, run command
```

The directory is `mjswan_app/`, **not** `mjswan/`: a `mjswan/` in the repo root shadows the installed package for every `python` invoked from there.

`main.py`, the shortest thing that works. `your_repo.tasks` stands for the registration module found in step 2, and `TASK_ID` is a real constant: substitute the value, never the name.

```python
"""Browser app for one mjlab task, built with mjswan."""

from __future__ import annotations

import json
import pathlib

import your_repo.tasks  # noqa: F401 - populates the mjlab task registry
import onnx

import mjswan

HERE = pathlib.Path(__file__).parent
TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"  # the task chosen in step 2


def setup_builder() -> mjswan.Builder:
    builder = mjswan.Builder()
    project = builder.add_project(name="G1 Velocity")
    scene = project.add_scene_mjlab(TASK_ID)

    meta = json.loads((HERE / "policy_meta.json").read_text())
    mdp = mjswan.MdpConfig()  # one MDP, traced once, shared by every checkpoint
    for name in meta.pop("checkpoints"):
        scene.add_policy(
            pathlib.Path(name).stem, onnx.load(HERE / name), mdp=mdp, **meta
        )
    return builder


def main() -> None:
    app = setup_builder().build()
    app.launch()


if __name__ == "__main__":
    main()
```

Canonical run command, from the repo root: `python -m mjswan_app.main`.

`build()` writes `mjswan_app/dist/`: the engine, plus the simulation document it serves — `manifest.json` and one `<project-id>/<scene-id>/` per scene holding `scene.mjz`, `mdp/<mdp-id>/{obs,term,command,event}/*.onnx` and `policy/<policy-id>.onnx`. `app.save_document()` packs that document alone as `dist.swn`, engine excluded, and `mjswan serve` / `info` / `publish` each take either form. Keep the `app` the build returns rather than chaining off it, so the document is one call away.

- W&B instead of local checkpoints: replace the loop with `scene.add_policy_wandb("entity/project/run_id")` and drop the meta plumbing.
- `add_policy` accepts the five MDP term sets — `observations=` / `actions=` / `terminations=` / `commands=` / `events=` — or one `mdp=mjswan.MdpConfig(...)` carrying all five, shared by every policy handed the same object. Leave them all out: `add_scene_mjlab` supplies `control_dt` and the trace env, each term set already defaults to the scene's env config (events included), and `add_policy_wandb` builds one `MdpConfig` per call so a run's checkpoints share one traced MDP. When looping over local checkpoints as above, build one `MdpConfig()` before the loop and pass `mdp=` to each `add_policy` for the same effect. To change a term set, mutate `env_cfg` instead (step 6) and pass `env_cfg=` to `add_scene_mjlab` *before* any policy is added, because a scene built first never sees the mutation.
- A single-input network — every mjlab export — needs no `in_keys`: its one observation group lands under the default slot, `actor`. A network with several inputs declares `in_keys=` on `add_policy`, and the build refuses one that does not. `out_keys=` is the same table for the *outputs*, and a network with several of them needs it whenever the action is not the first: the runtime drives the actuators from `out_keys`' `action`, defaulting to output 0. The build warns; a checkpoint whose exporter sidecar lists `out_keys` is telling you what to pass.
- When `terms.py` exists, import it in `main.py` for its side effects: `from mjswan_app import terms  # noqa: F401`.
- No `ViewerConfig`. The default `OriginType.AUTO` tracks the first non-world body.

## 6. Build, then fix what fails

```sh
python -m mjswan_app.main
```

The build fails loudly by design and names the ways out. For each failure: read the exception, read the module it came from, then classify.

- **Missing env-derived params**: mjlab's function needs a constant the config does not carry (terrain limits, a clip path, a threshold). Load `env_cfg` yourself, write the value into `env_cfg.<terms>[name].params`, and pass `env_cfg=` to `add_scene_mjlab`. This is the most common fix and needs no `terms.py` at all.
- **Untraceable body**: the function draws with an untraceable RNG, indexes per-env tensors, or reads state the browser has no equivalent for. Write a traceable equivalent in `terms.py` and register it:

  ```python
  # register_termination / register_event take the same two arguments.
  mjswan.register_observation("mjlab_func_name", my_traceable_func)
  ```

  Resolution goes by the mjlab function's `__name__` first, then the term's dict key, so a closure can only be reached by its key. The annotations on `register_event` / `register_termination` say `*Binding`, but the adapter accepts a plain traceable callable, which is what you want here; a `*Binding` means TypeScript and is out of scope.
- **Unbound command cfg class**: `mjswan.register_command("CfgClassName", CommandBinding(...))`, keyed by the *class* name, not a function name. Read `command.py` in the installed package for which shape applies (traced `state_fields` / `command_field`, versus native `ts_name` + `serializer`).
- **`MotionCommandCfg` reset-jitter warning** (tracking tasks): fetch and adapt `examples/mjlab/defaults/commands/__init__.py` from the mjswan repo, never re-derive the quaternion math, because `run_command_parity` traces the override itself and a mistake there is invisible to the parity gate. If you cannot fetch it, leave the warning in place and report it.
- **Missing asset or credential**: stop and hand the user the exact command (`wandb login`, `hf auth login`, a licence to accept, an env var to set).

Rebuild after each fix. A term skipped because mjswan cannot express it, rather than because the task is unusual, goes to step 8.

When it finally builds, read the document it wrote before spending minutes on parity:

```sh
mjswan info mjswan_app/dist
```

One line per project, scene, MDP and policy. Three things to check, none of which a successful build says out loud: **one MDP per shared `MdpConfig`** (one per W&B run, one for the whole local-checkpoint loop — more than that means a policy was handed its own config and the same terms were traced once per checkpoint); a **non-zero graph count** on each MDP (zero means every term ended up native or skipped); and **every checkpoint present** as a policy. The same command reads a `dist.swn`, so it is also how you inspect a document someone hands you.

## 7. Parity gate

A successful build only proves every term *traced*. It does not prove the graph computes what mjlab computes. Run the numeric gate, the same harness mjswan's own CI runs over its reference tasks:

```sh
MUJOCO_GL=disable python - <<'PY'
from mjlab.tasks.registry import load_env_cfg
from mjswan.mjlab import resolve_runner_defaults
from mjswan.compile import run_parity
from mjswan.mjlab.env import build_mjlab_env
import your_repo.tasks
import mjswan_app.terms  # only if terms.py exists

TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"
cfg = load_env_cfg(TASK_ID, play=True)
cfg.scene.num_envs = 1
obs_groups = resolve_runner_defaults(TASK_ID).obs_groups or {}
groups = obs_groups.get("actor")
report = run_parity(
    build_mjlab_env(cfg),
    obs_group=groups[0] if groups else "actor",
    n_steps=16,
)
print(report.summary())
raise SystemExit(0 if report.passed else 1)
PY
```

A failing term is a real defect: the exported graph disagrees numerically with mjlab's own function. Treat it exactly like a build failure. A term reported as `native` with a note is **not** a failure; it has no graph by design.

## 8. When the gap is mjswan's, open a PR

Steps 3, 6 and 7 each end in a block the target repo cannot route around. Some of those blocks are mjswan's gap rather than the task's, and then the fix is a pull request against mjswan.

### Only generic capabilities go in

A capability earns a place in mjswan only if another task would hit the same gap. Decide it mechanically by finding where the thing that failed is defined:

```sh
python -c "
import your_repo.tasks
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg('Mjlab-Velocity-Flat-Unitree-G1', play=True)
print(type(cfg.actions['TERM_NAME']).__module__)
"
```

- Defined in `mjlab`, `myosuite`, or another published package → generic. Open the PR.
- Defined in the target repo → task-specific. It does **not** go into mjswan. Use the author-side escape hatch in `terms.py`; if that cannot work either, report it and stop.

### Take the configuration from mjlab

A feature that makes the author restate what mjlab already declares is the wrong design. mjlab's config is the source; mjswan reads it.

- A new action term: name the mjswan cfg class exactly as mjlab names its own, or add the mapping to `_ACTION_CLASS_ALIASES` in `adapters/mjlab_adapter.py`. `_adapt_action_cfg` then copies every matching dataclass field by itself and the port needs no extra argument.
- Anything the task's env config or runner config already carries (`env_cfg`, `resolve_runner_defaults`) is read from there, never restated at the call site.
- A keyword argument the author has to pass by hand is the last resort, not the first.

### Where the change goes

Find the neighbouring case in the mjswan tree first: each already carries a worked example of the thing being added. An unsupported action term is the one exception to the no-TypeScript rule at the top of this file, because action is permanently native (ADR 0005 §7) and needs an engine-side `controlType` branch as well as a cfg class; that prohibition covers per-port `ts_src`, not mjswan's own engine.

### Change discipline

- Follow mjswan's `AGENTS.md`.
- One generic unit test at `tests/test_<feature>.py`; `tests/test_muscle_action.py` is the precedent. Test the feature directly. Do not add a demo scenario, a golden file, or the target's task to `examples/`.
- `make format && make type && uv run pytest -m "not slow"` before pushing.

### Open it

```sh
gh repo fork ttktjmt/mjswan --clone /tmp/mjswan-pr
cd /tmp/mjswan-pr && git checkout -b <branch>
# implement, test
gh pr create --repo ttktjmt/mjswan --title "<title>" --body "<body>"
```

The body names the task and term that motivated the change, states which package defines that term (the generality evidence), and says what the change does not cover.

The port stays blocked until the PR lands. Finish every other part of it, and report this one as waiting on the PR, with its URL.

## 9. Report

Write `mjswan_app/README.md`: the target's source URL, the task id, prerequisites (credentials, env vars, data files), the run command, and how the build is handed on — `app.save_document()` writes `mjswan_app/dist.swn`, which `mjswan serve`, `mjswan info` and `mjswan publish` each accept in place of the directory.

Then tell the user, in this order:

1. the `mjswan info` tree, then terms traced and parity result (`report.summary()` verbatim if anything failed);
2. terms skipped, each with what unblocks it: an author-side `terms.py` replacement, a `ts_src` TypeScript class (out of scope here), or the mjswan PR you opened in step 8, with its URL;
3. anything you stopped on: dependency conflicts, missing credentials, task-specific gaps that mjswan must not absorb;
4. that browser behaviour is unverified by this pipeline: parity proves the graph matches mjlab, but mjlab integrates with `mujoco_warp` while the browser runs MuJoCo's WASM build, and a policy can behave differently under the two.
