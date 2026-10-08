"""Launch file for the complete Forge simulation system.

Launches:
- forge_simulator (simulation lifecycle + object state + reset)
- forge_controller (joint command interface)
- forge_state (robot joint state publisher)

Usage:
    ros2 launch forge_sim forge_launch.py
    ros2 launch forge_sim forge_launch.py scenario:=basic_workspace
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scenario_arg = DeclareLaunchArgument(
        "scenario",
        default_value="basic_workspace",
        description="Scenario name to load",
    )

    simulator_node = Node(
        package="forge_sim",
        executable="simulator_node",
        name="forge_simulator",
        parameters=[{"scenario": LaunchConfiguration("scenario")}],
        output="screen",
    )

    controller_node = Node(
        package="forge_control",
        executable="controller_node",
        name="forge_controller",
        output="screen",
    )

    state_node = Node(
        package="forge_state",
        executable="state_node",
        name="forge_state",
        output="screen",
    )

    return LaunchDescription([
        scenario_arg,
        simulator_node,
        controller_node,
        state_node,
    ])
