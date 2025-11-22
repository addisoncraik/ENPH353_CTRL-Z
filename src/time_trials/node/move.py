#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from find_center import find_course_center
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import find_center
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Mover:
    def __init__(self):
        rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.callback)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.score_tracker = rospy.Publisher('/score_tracker', String, queue_size=1)
        self.bridge = CvBridge()
        self.move = Twist()
        self.last_dx = 0
        self.last_dy = 0
        self.last_time = rospy.Time.now().to_sec()
        rospy.sleep(1.0)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        dx,dy,dz = find_course_center(cv_image)
        now = rospy.Time.now().to_sec()
        dt = now - self.last_time
        self.last_time = now

        # Avoid divide-by-zero on first callback
        if dt == 0:
            dt = 1e-6

        # Compute derivatives
        ddx = (dx - self.last_dx) / dt
        ddy = (dy - self.last_dy) / dt

        self.last_dx = dx
        self.last_dy = dy

        # Tune these gains
        Kp = 0.01  # proportional gain
        Kd = 0.001    # damping gain

        vx = Kp * dx - Kd * ddx
        vy = Kp * dy - Kd * ddy

        # Optional clamp
        vx = max(min(vx, 5), -5)
        vy = max(min(vy, 5), -5)

        rospy.loginfo(
        f"[Centering] dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f} | "
        f"vx={vx:.3f}, vy={vy:.3f} | "
        f"ddx={ddx:.3f}, ddy={ddy:.3f} | dt={dt:.4f}"
        )

        self.move.linear.x = -vy
        self.move.linear.y = -vx
        self.move.linear.z = dz
        self.pub.publish(self.move)
        rgb_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Draw direction arrow (OpenCV only)
        rgb_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_img.shape
        cx, cy = w // 2, h // 2
        ex, ey = cx + dx, cy + dy

        image_with_vector = rgb_img.copy()
        cv2.arrowedLine(
            image_with_vector,
            (cx, cy),
            (ex, ey),
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

        # Display live with OpenCV (non-blocking)
        cv2.imshow("Direction", image_with_vector)
        cv2.waitKey(1)

def main():
    print("Loaded from:", find_center.__file__)
    rospy.init_node('robot_controller')
    mover = Mover()
    rospy.spin()

if __name__ == '__main__':
    main()
