#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>    
#include <gazebo/common/common.hh>
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

namespace gazebo {
  class TwistFlyPlugin : public ModelPlugin {
  physics::ModelPtr model;
  event::ConnectionPtr updateConection;
  std::unique_ptr<ros::NodeHandle> nh;
  ros::Subscriber cmdVelSubscriber;
  std::string robot_namespace_;
  
  ignition::math::Vector3d linearCmd{0,0,0};
  ignition::math::Vector3d angularCmd{0,0,0};
  
public: 
  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) {
    model = _model;
    
    // Checks for a namespace tag so that multiple robots can use this plugin
    robot_namespace_ = "";
    if (!_sdf->HasElement("robotNamespace")) {
    ROS_INFO("GazeboRosSkidSteerDrive Plugin missing <robotNamespace>, defaults to \"%s\"",
    this->robot_namespace_.c_str());
    } else {
    this->robot_namespace_ =
    _sdf->GetElement("robotNamespace")->Get<std::string>() + "/";
    }
    
    // Initialze the node and subscribe to cmd_vel
    // The namespace is just set to ~
    nh = std::make_unique<ros::NodeHandle>(robot_namespace_);
    // Not sure why I need the node handler to create the subscriber
    // Sets up a ros subcriber node so that the cmd_vel can be used
    cmdVelSubscriber = nh->subscribe("/cmd_vel",1,  &TwistFlyPlugin::OnCmdVel, this);
    
    // Updates the plugin each time the world is updated.
    // bind 
    updateConection = event::Events::ConnectWorldUpdateBegin(
      std::bind(&TwistFlyPlugin::OnUpdate, this)
    );
  }
  
  // Callback function for each time a cmd velocity is given
  void OnCmdVel(const geometry_msgs::Twist::ConstPtr &msg) {
    // Store the cmds in vectors. They are then applied when the simulation is updated next
    linearCmd = ignition::math::Vector3d(msg->linear.x, msg->linear.y, msg->linear.z);
    angularCmd = ignition::math::Vector3d(msg->angular.x, msg->angular.y, msg->angular.z);
  }

  void OnUpdate() {
    // Uses world coordinates to move the drone around may not be applicable for competition
    auto rot = model->WorldPose().Rot();
    ignition::math::Vector3d worldLinear = rot.RotateVector(linearCmd);
    model->SetLinearVel(worldLinear);
    model->SetAngularVel(angularCmd);
    //TODO: Apply a force on the model to counteract gravity so that we can hover
  }
};

GZ_REGISTER_MODEL_PLUGIN(TwistFlyPlugin);
}