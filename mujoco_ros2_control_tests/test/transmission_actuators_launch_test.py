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
Test of export_transmission_actuators: the actuators behind the transmissions are exported next to the joints.

joint1 drives actuator1 through a SimpleTransmission (mechanical_reduction 2.0); joint2 and joint3 drive actuator2
and actuator3 through a DifferentialTransmission. Either side can be commanded, the joint states keep coming from
the transmissions, and switches that would command both sides of a transmission, or only part of its joints, are
refused.
"""

import os
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from controller_manager.test_utils import check_controllers_running, check_node_running
from controller_manager_msgs.srv import ListHardwareInterfaces, SwitchController
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

MECHANICAL_REDUCTION = 2.0
TOLERANCE = 0.002
CONTROLLERS = [
    "joint1_position_controller",
    "actuator1_position_controller",
    "joint2_position_controller",
    "differential_position_controller",
]


@pytest.mark.rostest
def generate_test_description():
    launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("mujoco_ros2_control_tests"),
                "launch/transmission_actuators_test_launch.py",
            )
        ),
        launch_arguments={"headless": "true"}.items(),
    )

    return LaunchDescription([launch_include, KeepAliveProc(), ReadyToTest()])


class TestTransmissionActuators(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("transmission_actuators_test_node")
        self._latest_js = None
        self._latest_actuator_js = None
        self.node.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.node.create_subscription(JointState, "/mujoco_actuators_states", self._actuator_state_cb, 10)
        self.switch_client = self.node.create_client(SwitchController, "/controller_manager/switch_controller")
        check_controllers_running(self.node, ["joint_state_broadcaster"])
        check_controllers_running(self.node, CONTROLLERS, state="inactive")

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

    def switch(self, activate=(), deactivate=()):
        """Returns whether the controller manager accepted the switch."""
        self.assertTrue(self.switch_client.wait_for_service(timeout_sec=10.0), "switch_controller not available")
        request = SwitchController.Request()
        request.activate_controllers = list(activate)
        request.deactivate_controllers = list(deactivate)
        request.strictness = SwitchController.Request.STRICT
        future = self.switch_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        self.assertIsNotNone(future.result(), f"No answer to the switch {activate} / {deactivate}")
        return future.result().ok

    def command(self, controller, data, duration=2.0):
        pub = self.node.create_publisher(Float64MultiArray, f"/{controller}/commands", 10)
        self.assertTrue(
            self.spin_until(lambda: pub.get_subscription_count() > 0, timeout=5.0),
            f"{controller} did not subscribe to commands",
        )
        msg = Float64MultiArray(data=data)
        end_time = time.time() + duration
        while time.time() < end_time:
            pub.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def assert_reaches(self, msg_attr, name, expected):
        reached = self.spin_until(
            lambda: self.get_joint_value(getattr(self, msg_attr), "position", name) is not None
            and abs(self.get_joint_value(getattr(self, msg_attr), "position", name) - expected) < TOLERANCE,
            timeout=10.0,
        )
        value = self.get_joint_value(getattr(self, msg_attr), "position", name)
        self.assertTrue(reached, f"'{name}' is at {value}, expected {expected}")

    def test_node_start(self):
        check_node_running(self.node, "robot_state_publisher")

    def test_clock(self):
        with WaitForTopics([("/clock", Clock)], timeout=10.0):
            print("/clock is receiving messages!")

    def test_actuator_interfaces_exported(self):
        client = self.node.create_client(ListHardwareInterfaces, "/controller_manager/list_hardware_interfaces")
        self.assertTrue(client.wait_for_service(timeout_sec=10.0), "list_hardware_interfaces not available")
        future = client.call_async(ListHardwareInterfaces.Request())
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        self.assertIsNotNone(future.result(), "list_hardware_interfaces failed")

        state_interfaces = {iface.name for iface in future.result().state_interfaces}
        command_interfaces = {iface.name for iface in future.result().command_interfaces}
        for name in ["joint1", "joint2", "joint3", "actuator1", "actuator2", "actuator3"]:
            for interface in ["position", "velocity", "effort"]:
                self.assertIn(f"{name}/{interface}", state_interfaces)
            # MuJoCo position actuators: position commands only, on both sides of the transmissions
            self.assertIn(f"{name}/position", command_interfaces)
            self.assertNotIn(f"{name}/velocity", command_interfaces)
            self.assertNotIn(f"{name}/effort", command_interfaces)

    def test_actuator_commanded_directly(self):
        self.assertTrue(self.switch(activate=["actuator1_position_controller"]))
        try:
            # joint1 is not commanded, so its transmission leaves actuator1 to its own controller...
            self.command("actuator1_position_controller", [0.2])
            self.assert_reaches("_latest_actuator_js", "actuator1", 0.2)
            # ...while still reporting the joint state.
            self.assert_reaches("_latest_js", "joint1", 0.2 / MECHANICAL_REDUCTION)
        finally:
            self.assertTrue(self.switch(deactivate=["actuator1_position_controller"]))

    def test_joint_commanded_through_transmission(self):
        self.assertTrue(self.switch(activate=["joint1_position_controller"]))
        try:
            self.command("joint1_position_controller", [-0.05])
            self.assert_reaches("_latest_actuator_js", "actuator1", -0.05 * MECHANICAL_REDUCTION)
            self.assert_reaches("_latest_js", "joint1", -0.05)
        finally:
            self.assertTrue(self.switch(deactivate=["joint1_position_controller"]))

        # A controller taking actuator1 over starts from the command actuator1 got through the transmission, before
        # sending one of its own.
        self.assertTrue(self.switch(activate=["actuator1_position_controller"]))
        try:
            self.spin_until(lambda: False, timeout=1.0)
            actuator_position = self.get_joint_value(self._latest_actuator_js, "position", "actuator1")
            self.assertAlmostEqual(actuator_position, -0.05 * MECHANICAL_REDUCTION, delta=TOLERANCE)
        finally:
            self.assertTrue(self.switch(deactivate=["actuator1_position_controller"]))

    def test_joint_and_its_actuator_refused(self):
        self.assertTrue(self.switch(activate=["actuator1_position_controller"]))
        try:
            self.assertFalse(self.switch(activate=["joint1_position_controller"]))
        finally:
            self.assertTrue(self.switch(deactivate=["actuator1_position_controller"]))

        self.assertTrue(self.switch(activate=["joint1_position_controller"]))
        try:
            self.assertFalse(self.switch(activate=["actuator1_position_controller"]))
        finally:
            self.assertTrue(self.switch(deactivate=["joint1_position_controller"]))

    def test_partial_transmission_refused(self):
        # joint3 would feed NaN into the differential, so joint2 alone is refused...
        self.assertFalse(self.switch(activate=["joint2_position_controller"]))
        # ...and both joints together are accepted.
        self.assertTrue(self.switch(activate=["differential_position_controller"]))
        self.assertTrue(self.switch(deactivate=["differential_position_controller"]))
