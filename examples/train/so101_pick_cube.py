"""Train MuJoCo Playground's SO-101 PickCube policy for an mjswan scene.

MuJoCo Playground 1.0.1's SO-101 PickCube, trained with Playground's own environment and
PPO settings on one free GPU (Colab or molab), then exported for mjswan. `--task` picks
the variant:

  state   `SO101PickCube`: joints, gripper and cube positions, the target; 50 Hz
  vision  `SO101PickCubeVision`, the task its sim-to-real demo deploys: joints, gripper
          pose, the target in the gripper's frame and the wrist camera's image; 10 Hz

Four steps, each a subcommand:

  setup   install Playground 1.0.1 for this machine's CUDA driver
  train   PPO, resuming from the newest checkpoint in --out (or on the Hub)
  export  the policy as ONNX, its contract and a reference rollout (below)
  upload  --out's exports and the newest checkpoint to a Hugging Face model repository

`all` runs train, export and upload in order. Every step can be run again and picks up
from the newest checkpoint. A Colab or molab session takes its disk with it when it ends,
so pass --hf-repo there (a write token in HF_TOKEN): each checkpoint is also kept on the
Hub, and a session that starts with none on disk takes the Hub's newest.

`export` writes, under --out/export:

  policy.onnx     the actor in one file, its observation normalizer included:
                  obs [1, 24] -> actions [1, 6] (state), or obs [1, 33] and
                  camera [1, 3, 16, 16] -> actions [1, 6] (vision)
  contract.json   what the policy reads and drives, in order, for the mjswan port
  reference.npz   a deterministic rollout (no observation noise, no domain randomization):
                  each step's qpos, qvel, mocap_pos, ctrl, observations and action, and
                  for vision the wrist camera's 64 x 64 frame, so the port's observation
                  graphs can be checked against Playground's numbers

    python so101_pick_cube.py setup
    python so101_pick_cube.py all --out so101_pick_cube --hf-repo user/so101-pick-cube
    python so101_pick_cube.py all --task vision --out vision --hf-repo user/so101-vision
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

PLAYGROUND = "1.0.1"
TASKS = {"state": "SO101PickCube", "vision": "SO101PickCubeVision"}
#: Kept in a deterministic rollout: every training episode drew its geoms flat.
UNRANDOMIZED_TERMS = ("override_flat_materials",)
#: The reference rollout's length.
REFERENCE_SECONDS = 12.0
#: Warp 1.18.0's CUDA 12 build, for drivers that predate CUDA 13 (Colab's T4).
WARP_CU12_WHEELS = "https://github.com/NVIDIA/warp/releases/expanded_assets/v1.18.0"
TORCH_CU12_INDEX = "https://download.pytorch.org/whl/cu129"
EXTRAS = ["onnx", "onnxruntime", "huggingface_hub"]
CHECKPOINT = re.compile(r"model_(\d+)\.pt$")


# setup


def _driver_cuda() -> tuple[int, int] | None:
    """The CUDA version the driver supports, as `nvidia-smi` reports it."""
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    match = re.search(r"CUDA Version: (\d+)\.(\d+)", out)
    return (int(match[1]), int(match[2])) if match else None


def _installer() -> list[str]:
    """uv inside a virtual environment (molab's has no pip), pip for a system Python
    (Colab's), which uv may refuse to write to."""
    if sys.prefix != sys.base_prefix and shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "install"]


def setup(_: argparse.Namespace) -> None:
    if sys.version_info < (3, 12):
        sys.exit(
            f"Playground {PLAYGROUND} needs Python 3.12 or 3.13, not {sys.version}."
        )
    cuda = _driver_cuda()
    packages = [f"playground[rl]=={PLAYGROUND}", *EXTRAS]
    if cuda is None:
        print(
            "No NVIDIA driver: installing for CPU, enough to check the steps, not to train."
        )
        command = [*_installer(), *packages]
    elif cuda >= (13, 0):
        print(
            f"Driver supports CUDA {cuda[0]}.{cuda[1]}: installing the CUDA 13 build."
        )
        command = [*_installer(), *packages]
    else:
        print(
            f"Driver supports CUDA {cuda[0]}.{cuda[1]}: installing the CUDA 12 builds."
        )
        command = [
            *_installer(),
            "--extra-index-url",
            TORCH_CU12_INDEX,
            "--find-links",
            WARP_CU12_WHEELS,
            "warp-lang==1.18.0+cu12",
            *packages,
        ]
    subprocess.run(command, check=True)


# Shared


def _device() -> str:
    from mujoco_playground._src import xpu

    return xpu.resolve_device(None)


def _model_card(env_id: str) -> str:
    return f"""\
---
tags: [mujoco-playground, mjswan, robotics, reinforcement-learning]
---
# SO-101 PickCube for mjswan

PPO on MuJoCo Playground {PLAYGROUND}'s `{env_id}`, trained by mjswan's
`examples/train/so101_pick_cube.py`.

- `checkpoints/`: rsl-rl checkpoints, the newest resumes training
- `policy.onnx`, `contract.json`, `reference.npz`: the export an mjswan scene runs
"""


def _entry(env_id: str):
    from mujoco_playground._src.rl import env_registry

    return env_registry.get(env_id)


def _build(env_id: str, nworld: int, device: str, *, deterministic: bool):
    """Playground's own env for `env_id`, built as its registry builds it.

    `deterministic` drops the observation noise, the camera's pixel noise and the domain
    randomization, so a rollout's numbers depend on the state alone.
    """
    from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
    from reaf.simulation.torch import flat_scene

    entry = _entry(env_id)
    env_cfg = entry.env_cfg_fn()
    if deterministic:
        for group in env_cfg.observation_adapter.values():
            group.enable_noise = False
            for term in group.terms.values():
                if "pixel_noise_std" in term.params:
                    term.params["pixel_noise_std"] = 0.0
        dr = env_cfg.resetters.get("domain_randomization")
        if dr is not None:
            terms = dr.runtime_params["terms"]
            kept = {name: terms[name] for name in UNRANDOMIZED_TERMS if name in terms}
            if kept:
                dr.runtime_params["terms"] = kept
            else:
                env_cfg.resetters.pop("domain_randomization")
    env_cfg.sim.nworld = nworld
    env_cfg.sim.xpu_device = device
    env_cfg.sim.impl = "warp"
    env = flat_scene.build_env(env_cfg)
    runner_cfg = entry.runner_cfg
    runner_cfg.nworld = nworld
    if not device.startswith("cuda"):
        runner_cfg.torch_compile_mode = None
    wrapped = vecenv_wrapper.RslRlVecEnvWrapper(
        env, clip_actions=runner_cfg.clip_actions, cfg=runner_cfg.vecenv_wrapper_cfg
    )
    return env_cfg, runner_cfg, wrapped


def _iteration(path: Path) -> int:
    match = CHECKPOINT.search(path.name)
    return int(match[1]) if match else -1


def _newest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=_iteration)
    return checkpoints[-1] if checkpoints else None


def _hub():
    from huggingface_hub import HfApi

    return HfApi()


def _checkpoint(args: argparse.Namespace) -> Path | None:
    """The newest checkpoint on disk, else the Hub's, for a session that starts empty."""
    run_dir = args.out / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _newest_checkpoint(run_dir)
    if checkpoint is None and args.hf_repo:
        checkpoint = _pull_newest_checkpoint(args.hf_repo, run_dir)
    return checkpoint


def _pull_newest_checkpoint(repo: str, run_dir: Path) -> Path | None:
    from huggingface_hub import hf_hub_download

    names = [
        name
        for name in _hub().list_repo_files(repo)
        if name.startswith("checkpoints/") and CHECKPOINT.search(name)
    ]
    if not names:
        return None
    newest = max(names, key=lambda name: _iteration(Path(name)))
    print(f"Fetched {newest} from the Hub.")
    cached = hf_hub_download(repo, newest)
    target = run_dir / Path(newest).name
    shutil.copyfile(cached, target)
    return target


def _claim_repo(repo: str, env_id: str) -> None:
    """Create the repository and write its card, so a token that cannot write fails now
    rather than in the background on the first checkpoint."""
    api = _hub()
    api.create_repo(repo, exist_ok=True)
    api.upload_file(
        path_or_fileobj=_model_card(env_id).encode(),
        path_in_repo="README.md",
        repo_id=repo,
    )


def _push(repo: str, path: Path, path_in_repo: str) -> None:
    _hub().upload_file(
        path_or_fileobj=str(path), path_in_repo=path_in_repo, repo_id=repo
    )


def _nworld_for(device: str, default: int) -> int:
    """Playground's own count where the GPU holds it, at most 4096 on a 16 GB card."""
    import torch

    if not device.startswith("cuda"):
        return 16
    memory = torch.cuda.get_device_properties(torch.device(device)).total_memory
    return default if memory >= 40 * 2**30 else min(default, 4096)


# train


def train(args: argparse.Namespace) -> None:
    import dataclasses

    import warp as wp
    from mujoco_playground._src.rl import rng
    from rsl_rl.runners import OnPolicyRunner

    env_id = TASKS[args.task]
    defaults = _entry(env_id).runner_cfg
    iterations = args.iterations or defaults.max_iterations
    run_dir = args.out / "checkpoints"
    if args.hf_repo:
        _claim_repo(args.hf_repo, env_id)
    resume = _checkpoint(args)
    done = _iteration(resume) if resume else -1
    if done >= iterations - 1:
        print(f"Already trained: {resume.name}.")
        return

    device = _device()
    rng.seed(args.seed)
    wp.config.kernel_cache_dir = str(args.warp_cache)
    nworld = args.nworld or _nworld_for(device, defaults.nworld)
    print(f"Training {env_id} on {device} with {nworld} worlds, to {iterations}.")
    _, runner_cfg, env = _build(env_id, nworld, device, deterministic=False)
    for key, value in json.loads(args.overrides or "{}").items():
        setattr(runner_cfg, key, value)
    runner_cfg.seed = args.seed

    uploads: list[threading.Thread] = []
    failed: set[Path] = set()

    def push(saved: Path) -> None:
        try:
            _push(args.hf_repo, saved, f"checkpoints/{saved.name}")
        except Exception as error:  # The next checkpoint may still get there.
            failed.add(saved)
            print(f"Could not upload {saved.name} to the Hub: {error}", flush=True)

    class Runner(OnPolicyRunner):
        def save(self, path: str, infos: dict | None = None) -> None:
            super().save(path, infos)
            if args.hf_repo:
                # Off the training loop: the next iterations need not wait for the Hub.
                upload = threading.Thread(target=push, args=(Path(path),))
                upload.start()
                uploads.append(upload)

    runner = Runner(
        env, dataclasses.asdict(runner_cfg), log_dir=str(run_dir), device=device
    )
    if resume:
        # Actor, critic, optimizer and iteration: a fresh critic would undo the value fit.
        print(f"Resuming from {resume.name}.")
        runner.load(str(resume), map_location=device)
    # rsl-rl reruns the checkpoint's own iteration, so this ends on `iterations - 1`.
    runner.learn(
        num_learning_iterations=iterations - max(done, 0),
        init_at_random_ep_len=True,
    )
    env.close()
    for upload in uploads:
        upload.join()
    newest = _newest_checkpoint(run_dir)
    if newest in failed:
        # The last one is what the next session resumes from: this time a failure stops.
        _push(args.hf_repo, newest, f"checkpoints/{newest.name}")


# export


def _numpy(value):
    """A copy: on CPU the simulator's buffers would be aliased, every step the last."""
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.array(value.numpy(), copy=True)


def _jsonable(value):
    """A config value as JSON: sequences as lists, a function or class by its name."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return getattr(value, "__name__", repr(value))


def _contract(env_id: str, env_cfg, actor_groups, mj_model, control_dt: float) -> dict:
    """What the policy reads and drives, by name, in the order it reads them."""
    import mujoco

    devices = env_cfg.device_layer.devices
    resetter = env_cfg.resetters["pick_cube"].runtime_params

    def names(kind, count: int) -> list[str]:
        return [mujoco.mj_id2name(mj_model, kind, i) or "" for i in range(count)]

    def term(name: str, cfg) -> dict:
        entry = {
            "name": name,
            "shape": list(cfg.output_shape or []),
            "source": _jsonable(cfg.func),
            "params": _jsonable(cfg.params),
        }
        for key in ("scale", "offset", "clip"):
            if getattr(cfg, key) is not None:
                entry[key] = _jsonable(getattr(cfg, key))
        return entry

    contract = {
        "env": env_id,
        "playground": PLAYGROUND,
        "control_dt": control_dt,
        "physics_dt": float(mj_model.opt.timestep),
        "observations": {
            group: [
                term(name, cfg)
                for name, cfg in env_cfg.observation_adapter[group].terms.items()
            ]
            for group in actor_groups
        },
        "joints": list(devices["robot_state"].runtime_params["joint_names"]),
        "home_qpos": [
            float(v) for v in devices["robot_state"].runtime_params["home_qpos"]
        ],
        "action": {
            name: {
                "processor": component.component_type.__name__,
                "params": _jsonable(component.runtime_params),
            }
            for name, component in env_cfg.task_layer.command_processors.items()
        },
        "ee_site": devices["ee"].runtime_params["site_name"],
        "object_body": devices["object"].runtime_params["body_name"],
        "target_body": devices["target"].runtime_params["body_name"],
        "reset": {
            key: value
            for key, value in resetter.items()
            if key
            in (
                "object_pos_range",
                "target_pos_range",
                "resampling_time_range",
                "joint_perturbation",
            )
        },
        "model": {
            "qpos_joints": names(mujoco.mjtObj.mjOBJ_JOINT, mj_model.njnt),
            "jnt_qposadr": [int(a) for a in mj_model.jnt_qposadr],
            "jnt_dofadr": [int(a) for a in mj_model.jnt_dofadr],
            "actuators": names(mujoco.mjtObj.mjOBJ_ACTUATOR, mj_model.nu),
            "mocap_bodies": [
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b)
                for b in range(mj_model.nbody)
                if mj_model.body_mocapid[b] >= 0
            ],
        },
    }
    if "camera" in devices:
        camera = _jsonable(devices["camera"].runtime_params)
        name = camera["cam_name"][0]
        camera["fovy"] = float(mj_model.cam_fovy[mj_model.camera(name).id])
        contract["camera"] = camera
    return contract


def export(args: argparse.Namespace) -> None:
    import dataclasses

    import numpy as np
    import onnx
    import onnxruntime
    import torch
    import warp as wp
    from mujoco_playground._src.rl import rng
    from rsl_rl.runners import OnPolicyRunner

    env_id = TASKS[args.task]
    checkpoint = _checkpoint(args)
    if checkpoint is None:
        sys.exit("No checkpoint on disk or on the Hub: run `train` first.")
    export_dir = args.out / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    device = _device()
    rng.seed(args.seed)
    wp.config.kernel_cache_dir = str(args.warp_cache)
    env_cfg, runner_cfg, env = _build(env_id, 1, device, deterministic=True)
    runner = OnPolicyRunner(
        env, dataclasses.asdict(runner_cfg), log_dir=None, device=device
    )
    runner.load(
        str(checkpoint),
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
        },
        map_location=device,
    )
    runner.export_policy_to_onnx(str(export_dir), "policy.onnx")
    # torch writes the weights to a file of their own; a host loads one file.
    onnx_path = export_dir / "policy.onnx"
    onnx.save(onnx.load(str(onnx_path)), str(onnx_path))
    (export_dir / "policy.onnx.data").unlink(missing_ok=True)
    policy = runner.get_inference_policy(device)
    print(f"Exported {checkpoint.name} to {onnx_path}.")

    layer = env.unwrapped.device_layer
    sim = layer.sim
    control_dt = float(layer.control_timestep)
    steps = args.reference_steps or round(REFERENCE_SECONDS / control_dt)
    actor_groups = list(runner_cfg.obs_groups["actor"])
    obs = env.get_observations()
    # The network's flat input is its 1-D groups side by side; each image is an input.
    flat = [group for group in actor_groups if obs[group].ndim == 2]
    images = [group for group in actor_groups if obs[group].ndim > 2]
    camera_keys = [
        term.params["camera_key"]
        for group in images
        for term in env_cfg.observation_adapter[group].terms.values()
        if "camera_key" in term.params
    ]
    record: dict[str, list] = {
        key: []
        for key in ("qpos", "qvel", "mocap_pos", "ctrl", "obs", *images, "action")
    }
    for key in camera_keys:
        record[f"{key}_frame"] = []
    for _ in range(steps):
        with torch.inference_mode():
            action = policy(obs)
        record["qpos"].append(_numpy(sim.data.qpos)[0])
        record["qvel"].append(_numpy(sim.data.qvel)[0])
        record["mocap_pos"].append(_numpy(sim.data.mocap_pos)[0])
        record["ctrl"].append(_numpy(sim.data.ctrl)[0])
        record["obs"].append(np.concatenate([_numpy(obs[g])[0] for g in flat]))
        for group in images:
            record[group].append(_numpy(obs[group])[0])
        if camera_keys:
            frames = layer.get_device("camera").get_measurements(sim)
            for key in camera_keys:
                record[f"{key}_frame"].append(_numpy(frames[key])[0])
        record["action"].append(_numpy(action)[0])
        obs, _, _, _ = env.step(action)
    arrays = {
        key: np.asarray(
            values,
            dtype=np.uint8
            if key.endswith("_frame")
            else np.float32
            if key in images
            else np.float64,
        )
        for key, values in record.items()
    }
    np.savez_compressed(export_dir / "reference.npz", **arrays)

    # The ONNX graph must give the torch policy's actions on the same observations.
    session = onnxruntime.InferenceSession(str(onnx_path))
    inputs = session.get_inputs()
    onnx_actions = np.concatenate(
        [
            session.run(
                None,
                {i.name: arrays[i.name][t][None].astype(np.float32) for i in inputs},
            )[0]
            for t in range(steps)
        ]
    )
    gap = float(np.abs(onnx_actions - arrays["action"]).max())
    if gap > 1e-4:
        sys.exit(f"policy.onnx disagrees with the checkpoint by {gap:.2e}.")

    contract = _contract(env_id, env_cfg, actor_groups, sim.mj_model, control_dt)
    contract["inputs"] = [
        {
            "name": i.name,
            "groups": flat if i.name == "obs" else [i.name],
            "shape": list(i.shape),
        }
        for i in inputs
    ]
    contract["checkpoint"] = checkpoint.name
    contract["reference_steps"] = steps
    contract["onnx_vs_torch_max_abs"] = gap
    (export_dir / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"Wrote contract.json and reference.npz ({steps} steps).")
    env.close()


# upload


def upload(args: argparse.Namespace) -> None:
    if not args.hf_repo:
        print("No --hf-repo: nothing to upload. Everything is in", args.out)
        return
    export_dir = args.out / "export"
    if not (export_dir / "policy.onnx").exists():
        sys.exit("No export yet: run `export` first.")
    api = _hub()
    api.create_repo(args.hf_repo, exist_ok=True)
    api.upload_folder(folder_path=str(export_dir), repo_id=args.hf_repo)
    checkpoint = _newest_checkpoint(args.out / "checkpoints")
    if checkpoint:
        _push(args.hf_repo, checkpoint, f"checkpoints/{checkpoint.name}")
    print(f"Uploaded to https://huggingface.co/{args.hf_repo}")


def run_all(args: argparse.Namespace) -> None:
    train(args)
    export(args)
    upload(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("setup", help="install Playground for this machine").set_defaults(
        run=setup
    )
    for name, run in (
        ("train", train),
        ("export", export),
        ("upload", upload),
        ("all", run_all),
    ):
        step = steps.add_parser(name)
        step.set_defaults(run=run)
        step.add_argument(
            "--out", type=Path, required=True, help="where everything is kept"
        )
        step.add_argument(
            "--hf-repo", help="a Hugging Face model repository, user/name"
        )
        step.add_argument(
            "--task", choices=sorted(TASKS), default="state", help="which variant"
        )
        step.add_argument(
            "--iterations",
            type=int,
            help="default: Playground's own (1000 for state, 3000 for vision)",
        )
        step.add_argument(
            "--nworld", type=int, help="parallel worlds (default: by GPU memory)"
        )
        step.add_argument("--seed", type=int, default=0)
        step.add_argument(
            "--overrides", help='runner settings as JSON, e.g. {"save_interval": 50}'
        )
        step.add_argument(
            "--reference-steps",
            type=int,
            help=f"default: {REFERENCE_SECONDS:g} s of control steps",
        )
        step.add_argument(
            "--warp-cache",
            type=Path,
            default=Path("/tmp/wp_kernel_cache"),
            help="compiled kernels",
        )
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
