# Dependencies

What mjswan depends on, why each pin is what it is, and how new releases enter the lock.

## Python

Core: `mujoco==3.11.0`, `onnx>=1.20.0`, `nodeenv>=1.9.1`, `rich>=13.0.0`, `typer>=0.12.0`.
Tooling pins (the `check` extra): `ruff==0.16.0` and `ty==0.0.77` exactly, `pyright` floored.

Extras, one per asset source plus tooling: `wandb`, `hf` (`huggingface-hub`), `mjlab` (`mjlab==1.6.0` + `torch>=2.9.1`), `check` (ruff / ty / pyright), `dev` (`check` + every source + pytest / pre-commit), `demo` (`hf` + `mjlab` + `wbc-mjlab`: what `examples/demo/main.py` imports), `examples` (`wandb` + `demo` + `onnxruntime`). Each source extra is imported inside a function, so `import mjswan` touches none of them ([package-layout.md](package-layout.md)).

**`mjlab` + `torch` are build-time requirements for any policy carrying traced MDP terms**: tracing runs `torch.onnx.export` against a live mjlab env. Neither ships to the browser, and a model-only scene needs neither. `onnxruntime` (not `onnxruntime-web`) is the parity harness's runtime.

Python 3.11 to 3.13. 3.10 went at its end of life, when Pillow 13 and filelock 4.2 (the fixes for their advisories) required 3.11. The old `<3.13` cap was taken for `labmaze` (transitive via myosuite → dm-control, no cp313 wheel); myosuite left with the #128 examples cleanup, so the cap and the `h5py` override beside it are gone and the pytest matrix runs 3.13.

## MuJoCo, on both sides

The core `mujoco` pin is exact and follows mjlab's, which is what lets `mjswan[mjlab]` resolve at all: mjlab 1.6.0 requires `mujoco~=3.11.0` and brings `mujoco-warp` with it. Holding the core at 3.8.1 while mjlab moved to 3.10 and then 3.11 is what a `[tool.uv] override-dependencies` block once existed for; aligning the pin retired it, and with it a resolution that only ever worked inside this repository.

The browser's own engine is the template's `mujoco` dependency, an alias pinned to the `@ttktjmt/mujoco` release that holds the same MuJoCo: a `.mjz` carries MJCF the browser compiles itself, but a scene saved as `.mjb` loads only in the MuJoCo that wrote it. `tests/test_dependency_versions.py` holds the two pins together (a fork release that rebuilds a MuJoCo release, as 3.11.1 does 3.11.0, is listed there with the MuJoCo it holds) and `src/engine/__tests__/mujocoPackage.test.ts` checks the alias is what installs and runs that MuJoCo.

The alias is a fork because MuJoCo's own `@mujoco/mujoco` throws on every `mjtBool` array from 3.9.0 on; `@ttktjmt/mujoco` 3.11.1 is MuJoCo 3.11.0 with the generator fix of google-deepmind/mujoco#3616, built without debug info. The alias goes back to `@mujoco/mujoco` once a release carries the fix and mjlab's pin reaches that release.

## How releases enter the lock

- **Python**: `[tool.uv] exclude-newer = "3 days"` keeps a release out of the lock until it is three days old, the window in which hijacked releases have been caught and pulled. To take a younger one deliberately, name it in `[tool.uv] exclude-newer-package`; passed on the command line instead, it leaves a lock that `--locked` rejects. `ruff` is pinned exactly, not floored (see the comment in `pyproject.toml`).
- **npm**: the template's `.npmrc` refuses any package younger than three days (`min-release-age`), which a fresh fork release is. Its lockfile entry was written with that check off once, and `npm install` and `npm ci` then install from the lockfile without it. Pre-commit scans `.npmrc` for secrets.
