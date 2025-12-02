#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>    
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/Inertial.hh>
#include <gazebo/common/common.hh>
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <dynamic_reconfigure/server.h>
#include <twistflyplugin/TwistPIDConfig.h>


template<typename T> inline T clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo ? lo : (v > hi ? hi : v));
}

inline ignition::math::Vector3d clampVec(
    const ignition::math::Vector3d &v,
    double lo, double hi) {
    return ignition::math::Vector3d(
        clamp(v.X(), lo, hi),
        clamp(v.Y(), lo, hi),
        clamp(v.Z(), lo, hi)
    );
}


namespace gazebo {
  class TwistFlyPlugin : public ModelPlugin {
  physics::ModelPtr model;
  event::ConnectionPtr updateConection;
  std::unique_ptr<ros::NodeHandle> nh;
  ros::Subscriber cmdVelSubscriber;
  std::string robot_namespace_;
  std::string robot_base_frame_;
  gazebo::physics::LinkPtr base_link;
  
  // PID Controller Internal Variables
  double mass;
  double Kp = 0.01, Kd = 0.001, Ki = 100.0, clamp = 1.0;
  common::Time lastTime;
  ignition::math::Vector3d integral_error{0,0,0};
  ignition::math::Vector3d prevError{0,0,0};
  ignition::math::Vector3d prevDError{0,0,0};
  ignition::math::Vector3d linearCmd{0,0,0};
  ignition::math::Vector3d angularCmd{0,0,0};

  // Real time tuning tools
  std::unique_ptr<dynamic_reconfigure::Server<twistflyplugin::TwistPIDConfig>> server;
  dynamic_reconfigure::Server<twistflyplugin::TwistPIDConfig>::CallbackType f;
  
public: 
  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) {
    model = _model;
    
    // Checks for a namespace tag so that multiple robots can use this plugin
    robot_namespace_ = "";
    if (!_sdf->HasElement("robotNamespace")) {
    ROS_INFO("TwistFlyPlugin missing <robotNamespace>, defaults to \"%s\"",
    this->robot_namespace_.c_str());
    } else {
    this->robot_namespace_ =
    _sdf->GetElement("robotNamespace")->Get<std::string>();
    }

    this->robot_base_frame_ = "base_footprint";
    if (!_sdf->HasElement("robotBaseFrame")) {
      ROS_WARN("TwistFlyPlugin (ns = %s) missing <robotBaseFrame>, defaults to \"%s\"",
          this->robot_namespace_.c_str(), this->robot_base_frame_.c_str());
    } else {
      this->robot_base_frame_ = _sdf->GetElement("robotBaseFrame")->Get<std::string>();
    }
    this->base_link = model->GetLink(this->robot_base_frame_);
    this->mass = base_link->GetInertial()->Mass();
    this->lastTime = model->GetWorld()->SimTime();
    
    // Initialze the node and subscribe to cmd_vel
    // The namespace is just set to
    nh = std::make_unique<ros::NodeHandle>(robot_namespace_);
    // Not sure why I need the node handler to create the subscriber
    // server = std::make_unique<dynamic_reconfigure::Server<twistflyplugin::TwistPIDConfig>>(*nh);

    // // Bind the callback
    // dynamic_reconfigure::Server<twistflyplugin::TwistPIDConfig>::CallbackType f =
    //     boost::bind(&TwistFlyPlugin::cfgCallback, this, _1, _2);

    // server->setCallback(f);
    // // Sets up a ros subcriber node so that the cmd_vel can be used
    cmdVelSubscriber = nh->subscribe("/" + robot_namespace_ + "/cmd_vel", 1,  &TwistFlyPlugin::OnCmdVel, this);
    
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
    auto rot = model->WorldPose().Rot();
    ignition::math::Vector3d cmdWorld = rot.RotateVector(this->linearCmd);
    ignition::math::Vector3d vWorld = model->WorldLinearVel();

    // -------- PID CONTROLLER -----------
    // Find error function - difference between target velocity and actual velocity
    ignition::math::Vector3d vErr = (cmdWorld - vWorld) * 100;

    // Time calculations for integral and derivative
    common::Time simTime = model->GetWorld()->SimTime();
    double dt = (simTime - lastTime).Double();
    this->lastTime = simTime;
    if (dt <= 0) return;

    // Error function derivative + a low pass filter
    double alpha = 0.8;
    ignition::math::Vector3d dvErr = (vErr - this->prevError) / dt;
    dvErr = this->prevDError * alpha + (1 - alpha) * dvErr;

    // Error function integral (with a leak)
    double leak = 1;
    this->integral_error = leak * this->integral_error + vErr * dt;

    // Clamp derivative and integral error function
    dvErr = clampVec(dvErr, -this->clamp, this->clamp);
    this->integral_error = clampVec(this->integral_error, -this->clamp, this->clamp);

    // Find the resulting force. Scale it by the mass of the object
    ignition::math::Vector3d force = (Kp * vErr + Ki * integral_error - Kd * dvErr) * mass;

    // Update previous error and its derivative
    this->prevError = vErr;
    this->prevDError = dvErr;

    // Apply the force and the anglular velocity
    this->base_link->AddForce(force);
    this->model->SetAngularVel(this->angularCmd);
    ROS_INFO_STREAM_THROTTLE(1,
      "Force: " << force <<
      ", WorldVel: " << model->WorldLinearVel() <<
      ", commandVel: " << linearCmd <<
      ", Mass*g: " << mass*9.81
    );
  }

  void cfgCallback(twistflyplugin::TwistPIDConfig &config, uint32_t level){
    this->Kp = config.Kp;
    this->Kd = config.Kd;
    this->Ki = config.Ki;
    this->clamp = config.clamp;
    ROS_INFO("Reconfigure Request: kP = %f, kD = %f, kI = %f, clamp = %f", Kp, Kd, Ki, clamp);
  }
};

GZ_REGISTER_MODEL_PLUGIN(TwistFlyPlugin);
}