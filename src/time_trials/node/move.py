#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

rospy.init_node('topic_publisher')
cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
score_tracker = rospy.Publisher('/score_tracker', String, queue_size=1)

rate = rospy.Rate(0.1)
move_start = Twist()
move_start.linear.z = 0.5

move_end = Twist()
move_end.linear.x = 0

rate.sleep()
score_tracker.publish(str('TeamRed,multi21,0,xxxx'))
cmd_vel.publish(move_start)
rate.sleep()
cmd_vel.publish(move_end)
score_tracker.publish(str('TeamRed,multi21,-1,xxxx'))