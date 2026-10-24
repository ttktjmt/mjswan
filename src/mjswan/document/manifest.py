"""What the manifest is: its file name, the document format it declares, and the slot
defaults a policy entry may leave out (ADR 0006 §5, §7)."""

from __future__ import annotations

MANIFEST_NAME = "manifest.json"

#: The structure a build writes: what files exist, where they sit, and what the manifest
#: says about them. Bumped by hand, only when an engine reading the old structure would
#: misread the new one. Distinct from ``version``, the mjswan release: a host picks an
#: engine by ``version``, an engine protects itself by ``format`` (ADR 0006 §7).
#:
#: 2: ``input_slots`` may carry a raw ``mjData`` field (``{"sim": ...}``), narrowed to
#: ``rows``. A format-1 engine accepts the entry, cannot serve it, and freezes the
#: observation group at its previous value.
#:
#: 3: ``camera.azimuth`` is MuJoCo's, the direction the camera looks, not toward the
#: camera. An older engine would show every scene from the opposite side.
#:
#: 4: ``input_slots`` may carry the episode step counter (``{"env": ...}``), which
#: ``time_out`` is traced through. An older engine cannot serve it, so no episode would
#: ever time out.
#:
#: 5: ``input_slots`` may carry the policy's last action (``{"action": ...}``), which
#: ``last_action`` is traced through, or the rows of an ``mjModel`` field
#: (``{"model": ...}``), and an event may write one (``kind: "model"``), as model-field
#: randomization is traced. A command graph may take ``reset_mask`` and gate a write,
#: and a write may be a body-frame root velocity (``kind: "root_velocity_b"``). An older
#: engine cannot serve them: the observation group would freeze, the randomization would
#: not happen, and the command would not run.
#:
#: 6: an action term may be traced (``process`` and ``reset`` graphs, ADR 0009), carry
#: ``interpolate``, ``hold_s`` or ``delay``, or be ``joint_position_relative``, and an
#: input slot may read an action term's state (``{"action_term": ...}``). An older engine
#: would apply the raw action through the native formula, without any of them.
#:
#: 7: a scene may carry ``skins``, rigged characters drawn over the robot (``.vrm`` or
#: ``.glb`` under ``assets/``). An older engine would show the bare robot.
DOCUMENT_FORMAT = 7

#: Input slots the runtime fills itself rather than from an observation group: the
#: recurrent carry (``is_init``, ``adapt_hx``) and the step counter (``time_step``).
RUNTIME_INPUT_SLOTS = frozenset({"is_init", "adapt_hx", "time_step"})

#: What the runtime assumes when a policy declares no slot table (ADR 0006 §5).
DEFAULT_IN_KEYS = ("actor",)
DEFAULT_OUT_KEYS = ("action",)

__all__ = [
    "DEFAULT_IN_KEYS",
    "DEFAULT_OUT_KEYS",
    "DOCUMENT_FORMAT",
    "MANIFEST_NAME",
    "RUNTIME_INPUT_SLOTS",
]
