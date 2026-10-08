"""forge_controller — ROS 2 node that receives joint commands and applies them.

Subscribes to joint command messages on /forge/joint_commands and sets
MuJoCo actuator targets accordingly. Commands must NOT depend on
MuJoCo-specific code — they go through standard ROS interfaces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

_project_root = str(Path(__file__).resolve().parents[5])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.simulation import ForgeSimulation, PANDA_ALL_JOINTS


class ControllerNode(Node):
    """Receives joint commands and applies them to the simulation."""

    def __init__(self, sim: ForgeSimulation | None = None):
        super().__init__("forge_controller")

        self.declare_parameter("joint_command_topic", "/forge/joint_commands")
        cmd_topic = self.get_parameter("joint_command_topic").value

        # The simulation can be injected for in-process composition,
        # or this node can be used standalone with shared memory.
        self._sim = sim
        self._sim_lock = None

        # Subscribe to joint commands
        self._cmd_sub = self.create_subscription(
            JointState, cmd_topic, self._on_command, 10
        )

        self.get_logger().info(f"forge_controller listening on {cmd_topic}")

    def set_sim(self, sim: ForgeSimulation, lock=None):
        """Inject simulation reference for in-process use."""
        self._sim = sim
        self._sim_lock = lock

    def _on_command(self, msg: JointState):
        """Handle incoming joint command."""
        if self._sim is None:
            self.get_logger().warn("No simulation connected to controller")
            return

        targets = np.array(msg.position, dtype=np.float64)
        self.get_logger().debug(f"robot_command_received: {len(targets)} joints")

        if self._sim_lock:
            with self._sim_lock:
                self._sim.set_joint_targets(targets)
        else:
            self._sim.set_joint_targets(targets)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
