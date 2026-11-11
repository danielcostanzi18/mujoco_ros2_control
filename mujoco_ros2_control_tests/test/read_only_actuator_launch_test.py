#!/usr/bin/env python3

# Copyright 2026 PAL Robotics S.L.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test: claiming a joint leaves the control mode of the read_only actuators of its transmission alone.

relative_joint (effort) is measured relative to the 'reference' position actuator, which its transmission only
reads; 'reference' is commanded through reference_joint (position). Activating the effort controller of
relative_joint after the position controller of reference_joint must not switch 'reference' to effort, or the
position command of reference_joint would no longer reach it.
"""

import os
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from controller_manager.test_utils import check_controllers_running, check_node_running
from controller_manager_msgs.srv import SwitchController
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_testing.actions import ReadyToTest
from launch_testing.util import KeepAliveProc
from launch_testing_ros import WaitForTopics
import pytest
import rclpy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

REFERENCE_TARGET = 0.1
TOLERANCE = 0.002


@pytest.mark.rostest
def generate_test_description():
    launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("mujoco_ros2_control_tests"),
                "launch/read_only_actuator_test_launch.py",
            )
        ),
        launch_arguments={"headless": "true"}.items(),
    )

    return LaunchDescription([launch_include, KeepAliveProc(), ReadyToTest()])


class TestReadOnlyTransmissionActuator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("read_only_actuator_test_node")
        self._latest_js = None
        self._latest_actuator_js = None
        self.node.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.node.create_subscription(JointState, "/mujoco_actuators_states", self._actuator_state_cb, 10)
        self.switch_client = self.node.create_client(SwitchController, "/controller_manager/switch_controller")

    def tearDown(self):
        self.node.destroy_node()

    def _joint_state_cb(self, msg):
        self._latest_js = msg

    def _actuator_state_cb(self, msg):
        self._latest_actuator_js = msg

    def spin_until(self, predicate, timeout=15.0, spin_period=0.05):
        end_time = time.time() + timeout
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=spin_period)
            if predicate():
                return True
        return False

    def get_joint_value(self, msg, field, joint_name):
        if msg is None or joint_name not in msg.name:
            return None
        return getattr(msg, field)[msg.name.index(joint_name)]

    def activate(self, controller):
        self.assertTrue(self.switch_client.wait_for_service(timeout_sec=10.0), "switch_controller not available")
        request = SwitchController.Request()
        request.activate_controllers = [controller]
        request.strictness = SwitchController.Request.STRICT
        future = self.switch_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        self.assertIsNotNone(future.result(), f"No answer when activating '{controller}'")
        self.assertTrue(future.result().ok, f"Could not activate '{controller}'")

    def test_node_start(self):
        check_node_running(self.node, "robot_state_publisher")

    def test_clock(self):
        with WaitForTopics([("/clock", Clock)], timeout=10.0):
            print("/clock is receiving messages!")

    def test_joint_claim_keeps_read_only_actuator_mode(self):
        check_controllers_running(self.node, ["joint_state_broadcaster"])
        check_controllers_running(
            self.node, ["reference_position_controller", "relative_effort_controller"], state="inactive"
        )

        # The order matters: the effort claim on relative_joint comes last, so it would win the mode of 'reference'.
        self.activate("reference_position_controller")
        self.activate("relative_effort_controller")

        reference_pub = self.node.create_publisher(Float64MultiArray, "/reference_position_controller/commands", 10)
        relative_pub = self.node.create_publisher(Float64MultiArray, "/relative_effort_controller/commands", 10)
        self.assertTrue(
            self.spin_until(
                lambda: reference_pub.get_subscription_count() > 0 and relative_pub.get_subscription_count() > 0,
                timeout=5.0,
            ),
            "Controllers did not subscribe to commands",
        )

        reference_command = Float64MultiArray(data=[REFERENCE_TARGET])
        relative_command = Float64MultiArray(data=[0.0])
        end_time = time.time() + 2.0
        while time.time() < end_time:
            reference_pub.publish(reference_command)
            relative_pub.publish(relative_command)
            rclpy.spin_once(self.node, timeout_sec=0.05)

        # 'reference' is still position controlled, so the command of reference_joint reaches it.
        reached = self.spin_until(
            lambda: self.get_joint_value(self._latest_actuator_js, "position", "reference") is not None
            and abs(self.get_joint_value(self._latest_actuator_js, "position", "reference") - REFERENCE_TARGET)
            < TOLERANCE,
            timeout=10.0,
        )
        reference_position = self.get_joint_value(self._latest_actuator_js, "position", "reference")
        self.assertTrue(reached, f"MuJoCo actuator 'reference' is at {reference_position}, expected {REFERENCE_TARGET}")

        # ...and the transmission of relative_joint still reads it: with no effort, 'driven' stays at 0.
        relative_position = self.get_joint_value(self._latest_js, "position", "relative_joint")
        self.assertIsNotNone(relative_position, "No joint state received for relative_joint")
        self.assertAlmostEqual(
            relative_position,
            -REFERENCE_TARGET,
            delta=TOLERANCE,
            msg=f"relative_joint is at {relative_position}, expected {-REFERENCE_TARGET}",
        )
