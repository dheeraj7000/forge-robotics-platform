"""forge_simulator — ROS 2 node that owns the MuJoCo simulation lifecycle.

Responsibilities:
- Load and step the MuJoCo simulation
- Publish object ground-truth state
- Provide reset service
- Coordinate with forge_controller and forge_state via shared simulation
"""

from __future__ import annotations

import sys
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Quaternion, Point
from rclpy.node import Node
from std_msgs.msg import Header
from std_srvs.srv import Trigger

# Add forge project root to path so forge_core is importable
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[5])  # ros2_ws/src/forge_sim/forge_sim -> forge/
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.simulation import ForgeSimulation


class SimulatorNode(Node):
    """ROS 2 node wrapping the Forge MuJoCo simulation."""

    def __init__(self):
        super().__init__("forge_simulator")

        # Parameters
        self.declare_parameter("scenario", "basic_workspace")
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("object_state_topic", "/forge/object_states")
        self.declare_parameter("ee_pose_topic", "/forge/ee_pose")
        self.declare_parameter("reset_service", "/forge/reset")
        self.declare_parameter("sim_steps_per_publish", 10)

        scenario = self.get_parameter("scenario").value
        publish_rate = self.get_parameter("publish_rate").value
        obj_topic = self.get_parameter("object_state_topic").value
        ee_topic = self.get_parameter("ee_pose_topic").value
        reset_srv_name = self.get_parameter("reset_service").value
        self._sim_steps = self.get_parameter("sim_steps_per_publish").value

        # Initialize simulation
        self.get_logger().info(f"Loading scenario: {scenario}")
        self.sim = ForgeSimulation(scenario_name=scenario)
        self._sim_lock = threading.Lock()

        # Publishers
        self._object_pub = self.create_publisher(PoseArray, obj_topic, 10)
        self._ee_pub = self.create_publisher(PoseStamped, ee_topic, 10)

        # Reset service
        self._reset_srv = self.create_service(Trigger, reset_srv_name, self._handle_reset)

        # Timer for simulation stepping + publishing
        period = 1.0 / publish_rate
        self._timer = self.create_timer(period, self._tick)

        self.get_logger().info("forge_simulator ready")

    def _tick(self):
        """Step simulation and publish state."""
        with self._sim_lock:
            self.sim.step(self._sim_steps)
            self._publish_object_states()
            self._publish_ee_pose()

    def _publish_object_states(self):
        """Publish ground-truth object poses."""
        objects = self.sim.get_object_states()
        msg = PoseArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        for obj in objects:
            pose = Pose()
            pose.position = Point(x=float(obj.position[0]),
                                  y=float(obj.position[1]),
                                  z=float(obj.position[2]))
            pose.orientation = Quaternion(w=float(obj.orientation[0]),
                                          x=float(obj.orientation[1]),
                                          y=float(obj.orientation[2]),
                                          z=float(obj.orientation[3]))
            msg.poses.append(pose)

        self._object_pub.publish(msg)

    def _publish_ee_pose(self):
        """Publish end-effector pose."""
        robot = self.sim.get_robot_state()
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position = Point(x=float(robot.ee_position[0]),
                                   y=float(robot.ee_position[1]),
                                   z=float(robot.ee_position[2]))
        msg.pose.orientation = Quaternion(w=float(robot.ee_orientation[0]),
                                           x=float(robot.ee_orientation[1]),
                                           y=float(robot.ee_orientation[2]),
                                           z=float(robot.ee_orientation[3]))
        self._ee_pub.publish(msg)

    def _handle_reset(self, request, response):
        """Handle reset service call."""
        self.get_logger().info("reset_requested via service")
        with self._sim_lock:
            try:
                self.sim.reset()
                response.success = True
                response.message = "Simulation reset to initial state"
                self.get_logger().info("reset_completed")
            except Exception as e:
                response.success = False
                response.message = str(e)
                self.get_logger().error(f"controller_error: reset failed: {e}")
        return response

    def get_sim(self) -> ForgeSimulation:
        """Provide access to simulation for sibling nodes (in-process composition)."""
        return self.sim

    def get_sim_lock(self) -> threading.Lock:
        """Return the simulation thread lock."""
        return self._sim_lock


def main(args=None):
    rclpy.init(args=args)
    node = SimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sim.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
