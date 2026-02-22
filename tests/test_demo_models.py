"""Test suite for validating MuJoCo model files used in main.py.

This test ensures all XML model files referenced in the demo application
can be loaded successfully using mujoco.MjSpec.from_file().
"""

import sys
from pathlib import Path

import mujoco
import pytest

# Base directory for demo assets
DEMO_DIR = Path(__file__).parent.parent / "examples" / "demo"


# List of all model paths used in main.py
MODEL_PATHS = [
    # mjswan Demo Project
    "assets/scene/mjswan/unitree_go2/scene.xml",
    "assets/scene/mjswan/unitree_go1/go1.xml",
    "assets/scene/mjswan/unitree_g1/scene.xml",
    # MuJoCo Menagerie Project
    "assets/scene/mujoco_menagerie/agilex_piper/scene.xml",
    "assets/scene/mujoco_menagerie/agility_cassie/scene.xml",
    "assets/scene/mujoco_menagerie/aloha/scene.xml",
    "assets/scene/mujoco_menagerie/anybotics_anymal_b/scene.xml",
    "assets/scene/mujoco_menagerie/anybotics_anymal_c/scene.xml",
    "assets/scene/mujoco_menagerie/apptronik_apollo/scene.xml",
    "assets/scene/mujoco_menagerie/arx_l5/scene.xml",
    "assets/scene/mujoco_menagerie/berkeley_humanoid/scene.xml",
    "assets/scene/mujoco_menagerie/bitcraze_crazyflie_2/scene.xml",
    "assets/scene/mujoco_menagerie/booster_t1/scene.xml",
    "assets/scene/mujoco_menagerie/boston_dynamics_spot/scene.xml",
    "assets/scene/mujoco_menagerie/dynamixel_2r/scene.xml",
    "assets/scene/mujoco_menagerie/flybody/scene.xml",
    "assets/scene/mujoco_menagerie/fourier_n1/scene.xml",
    "assets/scene/mujoco_menagerie/franka_emika_panda/scene.xml",
    "assets/scene/mujoco_menagerie/franka_fr3/scene.xml",
    "assets/scene/mujoco_menagerie/google_barkour_v0/scene.xml",
    "assets/scene/mujoco_menagerie/google_barkour_vb/scene.xml",
    "assets/scene/mujoco_menagerie/google_robot/scene.xml",
    "assets/scene/mujoco_menagerie/hello_robot_stretch/scene.xml",
    "assets/scene/mujoco_menagerie/hello_robot_stretch_3/scene.xml",
    "assets/scene/mujoco_menagerie/i2rt_yam/scene.xml",
    "assets/scene/mujoco_menagerie/iit_softfoot/scene.xml",
    "assets/scene/mujoco_menagerie/kinova_gen3/scene.xml",
    "assets/scene/mujoco_menagerie/kuka_iiwa_14/scene.xml",
    "assets/scene/mujoco_menagerie/leap_hand/scene_left.xml",
    "assets/scene/mujoco_menagerie/leap_hand/scene_right.xml",
    "assets/scene/mujoco_menagerie/low_cost_robot_arm/scene.xml",
    "assets/scene/mujoco_menagerie/pal_talos/scene_motor.xml",
    "assets/scene/mujoco_menagerie/pal_talos/scene_position.xml",
    "assets/scene/mujoco_menagerie/pal_tiago/scene_motor.xml",
    "assets/scene/mujoco_menagerie/pal_tiago/scene_position.xml",
    "assets/scene/mujoco_menagerie/pal_tiago/scene_velocity.xml",
    "assets/scene/mujoco_menagerie/pal_tiago_dual/scene_motor.xml",
    "assets/scene/mujoco_menagerie/pal_tiago_dual/scene_position.xml",
    "assets/scene/mujoco_menagerie/pal_tiago_dual/scene_velocity.xml",
    "assets/scene/mujoco_menagerie/pndbotics_adam_lite/scene.xml",
    "assets/scene/mujoco_menagerie/rethink_robotics_sawyer/scene.xml",
    "assets/scene/mujoco_menagerie/robot_soccer_kit/scene.xml",
    "assets/scene/mujoco_menagerie/robotiq_2f85/scene.xml",
    "assets/scene/mujoco_menagerie/robotiq_2f85_v4/scene.xml",
    "assets/scene/mujoco_menagerie/robotis_op3/scene.xml",
    "assets/scene/mujoco_menagerie/shadow_hand/scene_left.xml",
    "assets/scene/mujoco_menagerie/shadow_hand/scene_right.xml",
    "assets/scene/mujoco_menagerie/skydio_x2/scene.xml",
    "assets/scene/mujoco_menagerie/stanford_tidybot/scene.xml",
    "assets/scene/mujoco_menagerie/tetheria_aero_hand_open/scene_right.xml",
    "assets/scene/mujoco_menagerie/trossen_vx300s/scene.xml",
    "assets/scene/mujoco_menagerie/trossen_wx250s/scene.xml",
    "assets/scene/mujoco_menagerie/trs_so_arm100/scene.xml",
    "assets/scene/mujoco_menagerie/ufactory_lite6/scene.xml",
    "assets/scene/mujoco_menagerie/ufactory_xarm7/scene.xml",
    "assets/scene/mujoco_menagerie/umi_gripper/scene.xml",
    "assets/scene/mujoco_menagerie/unitree_a1/scene.xml",
    "assets/scene/mujoco_menagerie/unitree_g1/scene.xml",
    "assets/scene/mujoco_menagerie/unitree_go1/scene.xml",
    "assets/scene/mujoco_menagerie/unitree_go2/scene.xml",
    "assets/scene/mujoco_menagerie/unitree_h1/scene.xml",
    "assets/scene/mujoco_menagerie/unitree_z1/scene.xml",
    "assets/scene/mujoco_menagerie/universal_robots_ur5e/scene.xml",
    "assets/scene/mujoco_menagerie/universal_robots_ur10e/scene.xml",
    "assets/scene/mujoco_menagerie/wonik_allegro/scene_left.xml",
    "assets/scene/mujoco_menagerie/wonik_allegro/scene_right.xml",
    # MuJoCo Playground Project - DeepMind Control Suite
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/acrobot.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/ball_in_cup.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/cartpole.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/cheetah.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/finger.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/fish.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/hopper.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/humanoid.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/manipulator.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/pendulum.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/point_mass.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/reacher.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/swimmer.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/walker.xml",
    # MuJoCo Playground Project - Manipulation Tasks
    "assets/scene/mujoco_playground/mujoco_playground/_src/manipulation/leap_hand/xmls/scene_mjx_cube.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/manipulation/franka_emika_panda/xmls/mjx_single_cube.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/manipulation/franka_emika_panda/xmls/mjx_single_cube_camera.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/manipulation/franka_emika_panda/xmls/mjx_cabinet.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/manipulation/aloha/xmls/mjx_hand_over.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/manipulation/aloha/xmls/mjx_single_peg_insertion.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/manipulation/franka_emika_panda_robotiq/xmls/scene_panda_robotiq_cube.xml",
    # MuJoCo Playground Project - Locomotion Tasks
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_flat_terrain.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_feetonly_flat_terrain.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_feetonly_bowl.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_feetonly_rough_terrain.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_feetonly_stairs.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_fullcollisions_flat_terrain.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/g1/xmls/scene_mjx_feetonly.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/g1/xmls/scene_mjx_feetonly_rough_terrain.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/h1/xmls/scene_mjx_feetonly.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/t1/xmls/scene_mjx_feetonly_flat_terrain.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/t1/xmls/scene_mjx_feetonly_rough_terrain.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/spot/xmls/scene_mjx_flat_terrain.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/spot/xmls/scene_mjx_feetonly_flat_terrain.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/apollo/xmls/scene_mjx_feetonly_flat_terrain.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/op3/xmls/scene_mjx_feetonly.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/berkeley_humanoid/xmls/scene_mjx_feetonly_flat_terrain.xml",
    "assets/scene/mujoco_playground/mujoco_playground/_src/locomotion/berkeley_humanoid/xmls/scene_mjx_feetonly_rough_terrain.xml",
    # MyoSuite Project
    "assets/scene/myosuite/myosuite/simhive/myo_sim/hand/myohand.xml",
    "assets/scene/myosuite/myosuite/simhive/myo_sim/arm/myoarm.xml",
    "assets/scene/myosuite/myosuite/simhive/myo_sim/elbow/myoelbow_2dof6muscles.xml",
    "assets/scene/myosuite/myosuite/simhive/myo_sim/leg/myolegs.xml",
    "assets/scene/myosuite/myosuite/simhive/myo_sim/finger/myofinger_v0.xml",
    "assets/scene/myosuite/myosuite/envs/myo/assets/arm/myoarm_relocate.xml",
    "assets/scene/myosuite/myosuite/envs/myo/assets/leg/myolegs_chasetag.xml",
    "assets/scene/myosuite/myosuite/envs/myo/assets/arm/myoarm_bionic_bimanual.xml",
    "assets/scene/myosuite/myosuite/envs/myo/assets/leg/myoosl_runtrack.xml",
    "assets/scene/myosuite/myosuite/envs/myo/assets/arm/myoarm_tabletennis.xml",
    "assets/scene/myosuite/myosuite/envs/myo/assets/leg_soccer/myolegs_soccer.xml",
]


@pytest.fixture
def demo_base_dir():
    """Fixture providing the base directory for demo assets."""
    return DEMO_DIR


@pytest.mark.parametrize("model_path", MODEL_PATHS)
def test_demo_model_loading(demo_base_dir, model_path):
    """Test that each MuJoCo model file can be loaded successfully.

    Args:
        demo_base_dir: Base directory containing the demo assets
        model_path: Relative path to the model XML file
    """
    full_path = demo_base_dir / model_path

    # Check if file exists
    assert full_path.exists(), f"Model file not found: {full_path}"

    # Try to load and compile the model (matches browser runtime behavior)
    try:
        spec = mujoco.MjSpec.from_file(str(full_path))
        assert spec is not None, f"Spec loaded but is None: {model_path}"
        model = spec.compile()
        assert model is not None, f"Spec compiled but model is None: {model_path}"
    except Exception as e:
        pytest.fail(f"Failed to load model {model_path}: {e}")


def test_all_model_paths_are_unique():
    """Verify that all model paths in the list are unique."""
    assert len(MODEL_PATHS) == len(set(MODEL_PATHS)), (
        "Duplicate model paths found in MODEL_PATHS"
    )


def test_demo_projects_count():
    """Test that main.py creates all expected scenes across all projects.

    This test validates:
    - All MODEL_PATHS are used to create scenes
    - The total number of scenes across all projects matches MODEL_PATHS
    """
    # Add the demo directory to sys.path to import the demo module
    demo_dir = Path(__file__).parent.parent / "examples" / "demo"
    sys.path.insert(0, str(demo_dir))

    try:
        from main import setup_builder  # pyright: ignore[reportMissingImports]  # noqa: I001

        builder = setup_builder()
        projects = builder.get_projects()

        # Count total number of scenes across all projects
        total_scenes = sum(len(project.scenes) for project in projects)

        assert total_scenes == len(MODEL_PATHS), (
            f"Expected {len(MODEL_PATHS)} scenes but found {total_scenes}"
        )
    finally:
        # Clean up sys.path
        sys.path.remove(str(demo_dir))
