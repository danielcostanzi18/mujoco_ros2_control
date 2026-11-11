// Copyright 2026 PAL Robotics S.L.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <transmission_interface/handle.hpp>
#include <transmission_interface/transmission.hpp>
#include <transmission_interface/transmission_loader.hpp>

namespace mujoco_ros2_control_tests
{

/**
 * @brief Test transmission: one joint measured relative to a reference actuator that it only reads.
 *
 * joint = driven - reference. The reference actuator is commanded through another transmission, so
 * joint_to_actuator() only writes the driven actuator, and the reference is meant to be declared read_only.
 */
class RelativeTransmission : public transmission_interface::Transmission
{
public:
  RelativeTransmission(std::string driven_name, std::string reference_name)
    : driven_name_(std::move(driven_name)), reference_name_(std::move(reference_name))
  {
  }

  void configure(const std::vector<transmission_interface::JointHandle>& joint_handles,
                 const std::vector<transmission_interface::ActuatorHandle>& actuator_handles) override
  {
    joint_position_ = find_handle(joint_handles, hardware_interface::HW_IF_POSITION);
    joint_velocity_ = find_handle(joint_handles, hardware_interface::HW_IF_VELOCITY);
    joint_effort_ = find_handle(joint_handles, hardware_interface::HW_IF_EFFORT);
    driven_position_ = find_handle(actuator_handles, hardware_interface::HW_IF_POSITION, driven_name_);
    driven_velocity_ = find_handle(actuator_handles, hardware_interface::HW_IF_VELOCITY, driven_name_);
    driven_effort_ = find_handle(actuator_handles, hardware_interface::HW_IF_EFFORT, driven_name_);
    reference_position_ = find_handle(actuator_handles, hardware_interface::HW_IF_POSITION, reference_name_);
    reference_velocity_ = find_handle(actuator_handles, hardware_interface::HW_IF_VELOCITY, reference_name_);
  }

  void actuator_to_joint() override
  {
    reference_position_value_ = reference_position_.get_value();
    reference_velocity_value_ = reference_velocity_.get_value();
    joint_position_.set_value(driven_position_.get_value() - reference_position_value_);
    joint_velocity_.set_value(driven_velocity_.get_value() - reference_velocity_value_);
    joint_effort_.set_value(driven_effort_.get_value());
  }

  // Uses the reference state of the last actuator_to_joint(): the reference's passthrough may already hold the command
  // of the transmission that drives it.
  void joint_to_actuator() override
  {
    driven_position_.set_value(joint_position_.get_value() + reference_position_value_);
    driven_velocity_.set_value(joint_velocity_.get_value() + reference_velocity_value_);
    driven_effort_.set_value(joint_effort_.get_value());
  }

  std::size_t num_actuators() const override
  {
    return 2;
  }

  std::size_t num_joints() const override
  {
    return 1;
  }

private:
  template <typename HandleT>
  static HandleT find_handle(const std::vector<HandleT>& handles, const std::string& interface_name,
                             const std::string& prefix_name = "")
  {
    const auto it = std::find_if(handles.begin(), handles.end(), [&](const HandleT& handle) {
      return handle.get_interface_name() == interface_name &&
             (prefix_name.empty() || handle.get_prefix_name() == prefix_name);
    });
    if (it == handles.end())
    {
      throw std::runtime_error("RelativeTransmission: no '" + interface_name + "' handle for '" + prefix_name + "'");
    }
    return *it;
  }

  std::string driven_name_;
  std::string reference_name_;
  double reference_position_value_ = 0.0;
  double reference_velocity_value_ = 0.0;

  transmission_interface::JointHandle joint_position_ = { "", "", nullptr };
  transmission_interface::JointHandle joint_velocity_ = { "", "", nullptr };
  transmission_interface::JointHandle joint_effort_ = { "", "", nullptr };
  transmission_interface::ActuatorHandle driven_position_ = { "", "", nullptr };
  transmission_interface::ActuatorHandle driven_velocity_ = { "", "", nullptr };
  transmission_interface::ActuatorHandle driven_effort_ = { "", "", nullptr };
  transmission_interface::ActuatorHandle reference_position_ = { "", "", nullptr };
  transmission_interface::ActuatorHandle reference_velocity_ = { "", "", nullptr };
};

/// Expects one joint and two actuators, with roles "driven" and "reference".
class RelativeTransmissionLoader : public transmission_interface::TransmissionLoader
{
public:
  std::shared_ptr<transmission_interface::Transmission>
  load(const hardware_interface::TransmissionInfo& transmission_info) override
  {
    auto actuator_with_role = [&transmission_info](const std::string& role) {
      const auto it =
          std::find_if(transmission_info.actuators.begin(), transmission_info.actuators.end(),
                       [&role](const hardware_interface::ActuatorInfo& actuator) { return actuator.role == role; });
      if (it == transmission_info.actuators.end())
      {
        throw std::runtime_error("RelativeTransmission '" + transmission_info.name + "' has no actuator with role '" +
                                 role + "'");
      }
      return it->name;
    };
    if (transmission_info.joints.size() != 1 || transmission_info.actuators.size() != 2)
    {
      throw std::runtime_error("RelativeTransmission '" + transmission_info.name +
                               "' needs exactly one joint and two actuators");
    }
    return std::make_shared<RelativeTransmission>(actuator_with_role("driven"), actuator_with_role("reference"));
  }
};

}  // namespace mujoco_ros2_control_tests

PLUGINLIB_EXPORT_CLASS(mujoco_ros2_control_tests::RelativeTransmissionLoader, transmission_interface::TransmissionLoader)
