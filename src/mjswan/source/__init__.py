"""Where policies and motion clips come from: a reference in, local files out.

A source knows how to talk to one service and nothing about what the files mean. That a
``model_*.pt`` needs mjlab's runner to become ONNX is :mod:`mjswan.mjlab.runner`'s
business, and attaching the result to a scene is the object model's. Each module here
backs one ``add_<layer>_<source>()`` family: ``source.wandb`` for ``add_policy_wandb`` /
``add_motion_wandb``, ``source.hf`` for ``add_policy_hf`` / ``add_motion_hf``.
"""

from . import hf, wandb

__all__ = ["hf", "wandb"]
