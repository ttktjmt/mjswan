"""Convert local mjlab ``.pt`` checkpoints to ONNX for mjswan.

Copied into a target repo as ``mjswan_app/export_policy.py`` by the
``mjlab-to-mjswan`` skill. Run from the repo root, with mjswan and the repo's own
task registrations importable from the same interpreter::

    python -m mjswan_app.export_policy TASK_ID logs/run/model_*.pt --register pkg.tasks

Writes one ``<stem>.onnx`` per checkpoint into ``--out-dir`` plus a
``policy_meta.json`` carrying what ``Scene.add_policy`` cannot recover from an ONNX
file alone -- ``policy_joint_names``, ``default_joint_pos``, ``encoder_bias`` -- and
the checkpoint order, numeric so ``model_50`` precedes ``model_100``.

Mirrors ``mjswan.mjlab.runner.export_checkpoint`` minus the W&B download,
``align_obs_normalizer`` included: without it the exported graph carries the wrong
observation normalization.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path


def _step(path: Path) -> int:
    """Training step from a ``model_<n>.pt`` name; 0 when the name carries no number."""
    match = re.search(r"\d+", path.stem)
    return int(match.group()) if match else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export mjlab .pt checkpoints to ONNX."
    )
    parser.add_argument("task_id", help="mjlab task id the checkpoints were trained on")
    parser.add_argument("checkpoints", nargs="+", type=Path, help="model_*.pt paths")
    parser.add_argument(
        "--register",
        action="append",
        metavar="MODULE",
        help="module to import first, to populate the mjlab task registry (repeatable)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    for module in args.register or []:
        importlib.import_module(module)

    import torch

    from mjswan.mjlab.runner import align_obs_normalizer, create_pt_onnx_export_context

    args.out_dir.mkdir(parents=True, exist_ok=True)
    context = create_pt_onnx_export_context(args.task_id)
    try:
        exported: list[str] = []
        for checkpoint in sorted(args.checkpoints, key=_step):
            onnx_name = f"{checkpoint.stem}.onnx"
            align_obs_normalizer(
                context.runner,
                torch.load(checkpoint, map_location="cpu", weights_only=False),
            )
            context.runner.load(
                str(checkpoint),
                load_cfg={"actor": True},
                strict=True,
                map_location="cpu",
            )
            context.runner.export_policy_to_onnx(str(args.out_dir), onnx_name)
            exported.append(onnx_name)
            print(f"wrote {args.out_dir / onnx_name}")

        meta = {
            "policy_joint_names": context.joint_names or None,
            "default_joint_pos": context.default_joint_pos or None,
            "encoder_bias": context.encoder_bias or None,
            "checkpoints": exported,
        }
    finally:
        context.close()

    meta_path = args.out_dir / "policy_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
