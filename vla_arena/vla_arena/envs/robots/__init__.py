from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda

from robosuite.robots import ROBOT_CLASS_MAPPING
import numpy as np
import os
import robosuite.utils.transform_utils as T

from copy import deepcopy
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.utils.transform_utils import mat2quat
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.models.base import MujocoModel
import robosuite.macros as macros
import robosuite.utils.transform_utils as T
from robosuite.controllers import controller_factory, load_part_controller_config
from robosuite.models.grippers import gripper_factory
from robosuite.robots.robot import Robot
from robosuite.robots import FixedBaseRobot
from robosuite.utils.buffers import DeltaBuffer, RingBuffer
from robosuite.utils.observables import Observable, sensor

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

TASK_MAPPING = {}


def register_problem(target_class):
    """We design the mapping to be case-INsensitive."""
    TASK_MAPPING[target_class.__name__.lower()] = target_class

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": FixedBaseRobot  ,
        "OnTheGroundPanda": FixedBaseRobot,
    }
)
