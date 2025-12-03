#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

class VelMux:
    def __init__(self):
        self.last_linear = Twist()
        self.last_angular = Twist()

        rospy.Subscriber("/Follower/cmd_vel_linear", Twist, self.linear_cb)
        rospy.Subscriber("/Follower/cmd_vel_angular", Twist, self.angular_cb)

        self.pub = rospy.Publisher("/Follower/cmd_vel", Twist, queue_size=10)

    def linear_cb(self, msg):
        self.last_linear = msg
        self.publish()

    def angular_cb(self, msg):
        self.last_angular = msg
        self.publish()

    def publish(self):
        cmd = Twist()
        cmd.linear = self.last_linear.linear
        cmd.angular = self.last_angular.angular
        self.pub.publish(cmd)

rospy.init_node("vel_mux")
VelMux()
rospy.spin()
