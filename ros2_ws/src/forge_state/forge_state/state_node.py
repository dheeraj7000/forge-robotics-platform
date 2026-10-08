"""forge_state — ROS 2 node that publishes robot joint state.

Publishes:
- sensor_msgs/JointState on /forge/joint_states (joint positions + velocities)
- geometry_msgs/PoseStamped on /forge/ee_pose (end-effector pose)

Uses standard ROS message types — no custom messages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

_project_root = str(Path(__file__).resolve().parents[5])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.simulation import ForgeSimulation


class StateNode(Node):
    """Publishes robot state from the simulation."""

    def __init__(self, sim: ForgeSimulation | None = None):
        super().__init__("forge_state")

        self.declare_parameter("joint_state_topic", "/forge/joint_states")
        self.declare_parameter("ee_pose_topic", "/forge/ee_pose")
        self.declare_parameter("publish_rate", 100.0)

        js_topic = self.get_parameter("joint_state_topic").value
        ee_topic = self.get_parameter("ee_pose_topic").value
        rate = self.get_parameter("publish_rate").value

        self._sim = sim
        self._sim_lock = None

        # Publishers
        self._js_pub = self.create_publisher(JointState, js_topic, 10)
        self._ee_pub = self.create_publisher(PoseStamped, ee_topic, 10)

        # Timer
        self._timer = self.create_timer(1.0 / rate, self._publish_state)

        self.get_logger().info(
            f"forge_state publishing on {js_topic} and {ee_topic} at {rate} Hz"
        )

    def set_sim(self, sim: ForgeSimulation, lock=None):
        """Inject simulation reference for in-process use."""
        self._sim = sim
        self._sim_lock = lock

    def _publish_state(self):
        if self._sim is None:
            return

        if self._sim_lock:
            with self._sim_lock:
                robot = self._sim.get_robot_state()
        else:
            robot = self._sim.get_robot_state()

        now = self.get_clock().now().to_msg()

        # Joint state
        js_msg = JointState()
        js_msg.header = Header(stamp=now, frame_id="world")
        js_msg.name = robot.joint_names
        js_msg.position = robot.joint_positions.tolist()
        js_msg.velocity = robot.joint_velocities.tolist()
        js_msg.effort = []  # not tracked in Phase 1
        self._js_pub.publish(js_msg)

        # End-effector pose
        ee_msg = PoseStamped()
        ee_msg.header = Header(stamp=now, frame_id="world")
        ee_msg.pose.position = Point(
            x=float(robot.ee_position[0]),
            y=float(robot.ee_position[1]),
            z=float(robot.ee_position[2]),
        )
        ee_msg.pose.orientation = Quaternion(
            w=float(robot.ee_orientation[0]),
            x=float(robot.ee_orientation[1]),
            y=float(robot.ee_orientation[2]),
            z=float(robot.ee_orientation[3]),
        )
        self._ee_pub.publish(ee_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
