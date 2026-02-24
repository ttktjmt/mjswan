"""mjswan: Browser-based MuJoCo Playground

Interactive MuJoCo simulations with ONNX policies running entirely in the browser.
"""

__version__ = "0.1.1"

from .app import mjswanApp
from .builder import Builder
from .command import (
    Button,
    ButtonConfig,
    CommandGroupConfig,
    CommandInput,
    Slider,
    SliderConfig,
    velocity_command,
)
from .policy import PolicyConfig, PolicyHandle
from .project import ProjectConfig, ProjectHandle
from .scene import SceneConfig, SceneHandle

__all__ = [
    # Builder and App
    "Builder",
    "mjswanApp",
    # Handles
    "ProjectHandle",
    "SceneHandle",
    "PolicyHandle",
    # Configs
    "ProjectConfig",
    "SceneConfig",
    "PolicyConfig",
    # Commands
    "Slider",
    "SliderConfig",
    "Button",
    "ButtonConfig",
    "CommandGroupConfig",
    "CommandInput",
    "velocity_command",
]
