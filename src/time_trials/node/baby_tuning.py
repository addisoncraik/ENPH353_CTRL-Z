#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import rospy
from baby_controller import BabyPID


def main():
    rospy.init_node('robot_controller')
    try:
        
        baby = BabyPID()
        #Start ScoreBoard
    except Exception as e:
        rospy.logerr("Failed to initialize Mover: %s", e)
        return
    rospy.spin()

if __name__ == '__main__':
    main()