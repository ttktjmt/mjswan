"""The mjlab metadata reader, tested against mjlab's own encoder.

``_list_to_csv_str`` below mirrors ``mjlab.rl.exporter_utils.list_to_csv_str``
verbatim, so these tests encode the way mjlab does and decode the way mjswan does.
Copied rather than imported, since the reader has to work without mjlab installed.
"""

import onnx
import pytest

from mjswan.mjlab.onnx_meta import action_cfg_from_metadata, read_mjlab_metadata


def _list_to_csv_str(arr, *, decimals=3, delimiter=",", sub_delimiter=";"):
    fmt = f"{{:.{decimals}f}}"

    def format_scalar(x):
        return fmt.format(x) if isinstance(x, (int, float)) else str(x)

    def format_entry(x):
        if isinstance(x, (list, tuple)):
            return sub_delimiter.join(format_scalar(v) for v in x)
        return format_scalar(x)

    return delimiter.join(format_entry(x) for x in arr)


def _model_with(metadata):
    """An ONNX model carrying ``metadata`` the way ``attach_metadata_to_onnx`` writes."""
    model = onnx.ModelProto()
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = _list_to_csv_str(value) if isinstance(value, list) else str(value)
    return model


#: A two-joint export, small enough to assert on in full.
BASE_METADATA = {
    "run_path": "brisk-cloud-42",
    "joint_names": ["hip", "knee"],
    "joint_stiffness": [40.0, 40.0],
    "joint_damping": [1.0, 1.0],
    "default_joint_pos": [-0.312, 0.669],
    "command_names": ["velocity"],
    "observation_names": ["base_lin_vel", "joint_pos"],
    "observation_terms_scale": [1.0, [0.5, 0.25]],
    "observation_terms_flatten_history_dim": [True, False],
    "observation_terms_history_length": [0, 3],
    "observation_terms_clip": [[float("-inf"), float("inf")], [-100.0, 100.0]],
    "action_scale": [0.5, 0.25],
}


class TestReadMjlabMetadata:
    def test_returns_none_without_mjlab_keys(self):
        assert read_mjlab_metadata(_model_with({"mjswan.kind": "observation"})) is None

    def test_returns_none_for_a_bare_model(self):
        assert read_mjlab_metadata(onnx.ModelProto()) is None

    def test_reads_joint_names_and_rest_pose(self):
        meta = read_mjlab_metadata(_model_with(BASE_METADATA))
        assert meta is not None
        assert meta.joint_names == ["hip", "knee"]
        assert meta.default_joint_pos == [-0.312, 0.669]
        assert meta.run_path == "brisk-cloud-42"

    def test_reads_gains(self):
        meta = read_mjlab_metadata(_model_with(BASE_METADATA))
        assert meta.joint_stiffness == [40.0, 40.0]
        assert meta.joint_damping == [1.0, 1.0]

    def test_history_lengths_survive_the_float_formatter(self):
        """mjlab writes ints through `{:.3f}`, so `3` arrives as `"3.000"`."""
        meta = read_mjlab_metadata(_model_with(BASE_METADATA))
        assert meta.observation_terms_history_length == [0, 3]

    def test_bools_survive_the_float_formatter(self):
        """`isinstance(True, int)` is true in Python, so mjlab writes `"1.000"`."""
        meta = read_mjlab_metadata(_model_with(BASE_METADATA))
        assert meta.observation_terms_flatten_history_dim == [True, False]

    def test_per_dimension_scale_reads_as_a_vector(self):
        meta = read_mjlab_metadata(_model_with(BASE_METADATA))
        assert meta.observation_terms_scale == [1.0, [0.5, 0.25]]

    def test_infinite_clip_bounds_round_trip(self):
        meta = read_mjlab_metadata(_model_with(BASE_METADATA))
        assert meta.observation_terms_clip == [
            (float("-inf"), float("inf")),
            (-100.0, 100.0),
        ]

    def test_empty_command_list_reads_as_empty(self):
        meta = read_mjlab_metadata(_model_with({**BASE_METADATA, "command_names": []}))
        assert meta.command_names == []

    def test_scalar_action_scale_keeps_full_precision(self):
        """Only lists go through the formatter; a scalar goes through `str()`."""
        meta = read_mjlab_metadata(
            _model_with({**BASE_METADATA, "action_scale": 0.0625})
        )
        assert meta.action_scale == 0.0625

    def test_tracking_fields(self):
        meta = read_mjlab_metadata(
            _model_with(
                {
                    **BASE_METADATA,
                    "anchor_body_name": "torso_link",
                    "body_names": ["torso_link", "left_foot"],
                }
            )
        )
        assert meta.anchor_body_name == "torso_link"
        assert meta.body_names == ["torso_link", "left_foot"]

    def test_raw_keeps_unknown_keys(self):
        meta = read_mjlab_metadata(_model_with({**BASE_METADATA, "task_extra": "x"}))
        assert meta.raw["task_extra"] == "x"


class TestActionCfgFromMetadata:
    def test_builds_a_joint_position_term(self):
        meta = read_mjlab_metadata(_model_with(BASE_METADATA))
        actions = action_cfg_from_metadata(meta, num_actions=2)
        assert list(actions) == ["joint_pos"]
        term = actions["joint_pos"]
        assert term.scale == [0.5, 0.25]
        assert term.use_default_offset is True

    def test_carries_a_scalar_scale_through(self):
        meta = read_mjlab_metadata(_model_with({**BASE_METADATA, "action_scale": 0.5}))
        assert action_cfg_from_metadata(meta, num_actions=2)["joint_pos"].scale == 0.5

    def test_a_passive_joint_does_not_count(self):
        """mjlab writes every joint but one scale per action, as for its YAM."""
        meta = read_mjlab_metadata(
            _model_with(
                {
                    **BASE_METADATA,
                    "joint_names": ["hip", "knee", "passive"],
                    "default_joint_pos": [-0.312, 0.669, 0.0],
                }
            )
        )
        assert action_cfg_from_metadata(meta, num_actions=2)["joint_pos"].scale == [
            0.5,
            0.25,
        ]

    def test_empty_without_a_recorded_scale(self):
        metadata = {k: v for k, v in BASE_METADATA.items() if k != "action_scale"}
        meta = read_mjlab_metadata(_model_with(metadata))
        assert action_cfg_from_metadata(meta, num_actions=2) == {}

    def test_empty_when_the_scale_is_not_one_per_action(self):
        meta = read_mjlab_metadata(_model_with(BASE_METADATA))
        assert action_cfg_from_metadata(meta, num_actions=12) == {}


class TestMalformedMetadata:
    def test_a_broken_clip_range_says_so(self):
        model = _model_with(BASE_METADATA)
        for entry in model.metadata_props:
            if entry.key == "observation_terms_clip":
                entry.value = "-inf;0;inf"
        with pytest.raises(ValueError, match="min;max"):
            read_mjlab_metadata(model)
