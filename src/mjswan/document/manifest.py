"""What the manifest is: its file name, the document format it declares, and the slot
defaults a policy entry may leave out."""

from __future__ import annotations

MANIFEST_NAME = "manifest.json"

#: The engine's directory in an app: its scripts, styles, WASM and the author's
#: custom-term module. Never part of a document, so no project may take it as its id.
ENGINE_DIR = "assets"

#: The structure a build writes: what files exist, where they sit, and what the manifest
#: says about them. Bumped by hand when a released engine that reads the current format
#: would ignore an addition and so run wrong; an addition to a format no release has
#: written yet joins it. Distinct from ``version``, the mjswan release: a host picks an
#: engine by ``version``, an engine protects itself by ``format``. Bump it with
#: ``MAX_DOCUMENT_FORMAT`` (``template/src/manifest/index.ts``) and add a row to the
#: format table in ``context/document.md``.
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
#: 6: an action term may be traced (``process`` and ``reset`` graphs), carry
#: ``interpolate``, ``hold_s`` or ``delay``, or be ``joint_position_relative``, and an
#: input slot may read an action term's state (``{"action_term": ...}``). An older engine
#: would apply the raw action through the native formula, without any of them.
#:
#: 7: a scene may carry ``skins``, rigged characters drawn over the robot (``.vrm`` or
#: ``.glb`` under ``assets/``). An older engine would show the bare robot.
#:
#: 8: a motion may carry ``frame_fields``, arrays of its ``.npz`` a tracking command's
#: ``{command, field}`` slot reads at the current frame. An older engine cannot serve
#: them, and the graph reading one would hold its previous value.
#:
#: 9: a graph's ``sensors`` may describe a camera (``kind: "camera"``), whose ``rgb`` the
#: engine renders, and an observation group may feed a policy input of more than two
#: dimensions (an image). A skin may be modelled on the robot (``fit: "modelled"``), each
#: of its joints fixed to the body it rests on, and may carry ``align``, ``rest_qpos``, a
#: solid ``material`` and ``hide_tendons``. A scene carries ``graphics``: its colour
#: management (MuJoCo's ``classic``, or ``filament``'s, with tone mapping and
#: exposure) and the shadow, reflection and haze switches, no longer in ``camera``. An
#: older engine would read the camera as a raycast sensor, retarget the skin as a
#: character, drawing it off the robot, and draw every scene its one way.
#:
#: 10: an action term may carry a DC motor's torque-speed curve (``saturation_effort``,
#: ``velocity_limit``). An older engine would run those motors without it.
#:
#: 11: a scene may carry ``term_defaults``, what its terms do when the viewer opens. An
#: older engine would start every term as mjlab's.
DOCUMENT_FORMAT = 11

#: Input slots the runtime fills itself rather than from an observation group: the
#: recurrent carry (``is_init``, ``adapt_hx``) and the step counter (``time_step``).
RUNTIME_INPUT_SLOTS = frozenset({"is_init", "adapt_hx", "time_step"})

#: What the runtime assumes when a policy declares no slot table.
DEFAULT_IN_KEYS = ("actor",)
DEFAULT_OUT_KEYS = ("action",)


def author_code(manifest: dict) -> bool:
    """Whether ``manifest`` asks for author code: a ``plugins`` module, whatever its
    value, or a true ``uses_custom_js``.

    A build writes both for custom-TS terms, whose module ships with the engine and never
    in a document, so a document that asks for one cannot be served or published.
    """
    return "plugins" in manifest or bool(manifest.get("uses_custom_js"))


__all__ = [
    "DEFAULT_IN_KEYS",
    "DEFAULT_OUT_KEYS",
    "DOCUMENT_FORMAT",
    "ENGINE_DIR",
    "MANIFEST_NAME",
    "RUNTIME_INPUT_SLOTS",
    "author_code",
]
