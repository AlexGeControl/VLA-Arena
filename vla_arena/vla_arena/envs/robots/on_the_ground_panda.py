import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OnTheGroundPanda(ManipulatorModel):
    """
    Panda is a sensitive single-arm robot designed by Franka.
    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """
    arms = ['right']
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/panda/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(
            attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
        )
        
    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return {"right": "PandaGripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_panda"}

    @property
    def init_qpos(self):
        return np.array([
            0,                    # 关节1: 基座旋转，保持0
            -1.61037389e-01,     # 关节2: 肩关节，保持原值
            0.00,                # 关节3: 保持0
            -2.8,                # 关节4: 肘关节，从-2.44改为-2.8（更弯曲，降低高度）
            0.00,                # 关节5: 保持0
            2.3,                 # 关节6: 腕关节，从2.23改为1.8（降低末端高度）
            np.pi / 4            # 关节7: 末端旋转，保持45度
        ])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, -0.7),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, -0.3),
            "living_room_table": lambda table_length: (
                -0.16 - table_length / 2,
                0,
                0.42,
            ),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
