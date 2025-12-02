#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


from dynamic_reconfigure.server import Server
from time_trials.cfg import CenteringPIDConfig

from isolate_map import isolate_map
from find_center import find_course_center
from find_clueboards import find_clue_boards

from find_babyDrone import find_babyDrone
from targeting import is_at_target
from targeting import find_target

from baby_controller import BabyPID

from read_boards.process_image import process_image

import constants as consts

class Mover:
    def __init__(self):
        self.Kp = 1.5
        self.Kd = 0.3
        self.Ki = 0.2
        self.imax = 10
        self.baby = BabyPID()
        rospy.Subscriber('/Follower/rrbot/camera1/image_raw', Image, self.baby_callback)
        rospy.Subscriber('/Master/rrbot/camera1/image_raw', Image, self.img_callback)
        rospy.Subscriber('/Master/rrbot/height', LaserScan, self.height_callback)
        self.pub = rospy.Publisher('/Master/cmd_vel', Twist, queue_size=1)
        self.baby_pub = rospy.Publisher('/Follower/cmd_vel', Twist, queue_size=1)
        self.score_tracker = rospy.Publisher('/score_tracker', String, queue_size=1)
        self.debug_pub = rospy.Publisher("/centering_debug", Vector3, queue_size=1)
        self.bridge = CvBridge()
        self.move = Twist()
        self.move_baby = Twist()
        self.boards = [
            [(500,250), [(490, 250), (510, 250)]]
            ]
        self.last_dx = 0
        self.last_dy = 0
        self.last_dz = 0
        self.integral_dx = 0
        self.integral_dy = 0
        self.dy_filtered = 0
        self.dx_filtered = 0
        self.height = -1

        self.prev_baby_location = (0,0,0)

        self.is_master_stable = False
        self.number_stable_frames = 0
        self.stable_baby_frames = 0
        self.baby_is_stable = False
        self.first_time_stable = True
        self.last_time = rospy.Time.now().to_sec()
        rospy.sleep(1.0)
        # self.srv = Server(CenteringPIDConfig, self.cfg_callback)

    def height_callback(self, data):
       valid_ranges = [r for r in data.ranges if r != float('inf')]

       if len(valid_ranges) > 0:
          avg_height = sum(valid_ranges) / len(valid_ranges)
          self.height = avg_height
       else:
          self.height = -1

    # def para_correct(self, pos):
    #    x,y,angle = pos

    #    cx = (consts.MAP_WIDTH/consts.SCALE_FACTOR)/2
    #    cy = (consts.MAP_HEIGHT/consts.SCALE_FACTOR)/2

    #    x_c = x - cx
    #    y_c = y - cy

    # #    if self.height == -1 or self.baby.height == -1:
    # #       k = 1
    # #    else:
    #    k = 0.97

    #    return x_c*k + cx, y_c*k + cy, angle

          
    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        self.stabilize_master(cv_image)
        
        if self.is_master_stable == True:
          map = isolate_map(cv_image, int(consts.MAP_WIDTH/consts.SCALE_FACTOR), int(consts.MAP_HEIGHT/consts.SCALE_FACTOR))
          if self.first_time_stable == True:
            self.first_time_stable = False
            self.boards = find_clue_boards(cv_image)
          rgb_map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
          cv2.imshow("map", rgb_map)
          babyDrone = find_babyDrone(map, self.prev_baby_location)
          board, target = find_target(babyDrone, self.boards)
          if board is None:
            return
          baby_vx, baby_vy, baby_angularz, baby_vz = self.baby.calculate_action(babyDrone, target, board, map)

          self.deleteTarget(board, target)
          self.move_baby.linear.x = baby_vx
          self.move_baby.linear.y = baby_vy 
          self.move_baby.angular.z = baby_angularz
          self.move_baby.linear.z = baby_vz
          self.baby_pub.publish(self.move_baby)
          self.prev_baby_location = babyDrone
        else:
          self.move_baby.linear.x = 0
          self.move_baby.linear.y = 0
          self.move_baby.angular.z = 0
          self.baby_pub.publish(self.move_baby)

    # def cfg_callback(self, config, level):
    #     self.Kp = config.Kp
    #     self.Kd = config.Kd
    #     self.Ki = config.Ki
    #     self.imax = config.imax
    #     return config
        
    def stabilize_master(self, cv_image):

        dx,dy,dz = find_course_center(cv_image)
        now = rospy.Time.now().to_sec()
        dt = now - self.last_time
        self.last_time = now

        # Avoid divide-by-zero on first callback
        if dt == 0:
            dt = 1e-6

        dx /= 100
        dy /= 100

        self.isMasterStable(dy, dx)

        # Compute derivatives
        ddx = (dx - self.last_dx) / dt
        ddy = (dy - self.last_dy) / dt
        ddz = (dz - self.last_dz) / dt
        alpha = 0.5
        self.dx_filtered = alpha * self.dx_filtered + (1-alpha) * ddx
        self.dy_filtered = alpha * self.dy_filtered + (1-alpha) * ddy

        self.dx_filtered = max(min(self.dx_filtered, self.imax), -self.imax)
        self.dy_filtered = max(min(self.dx_filtered, self.imax), -self.imax)
        ddz = max(min(ddz, self.imax), -self.imax)

        # # Compute Integrals
        # self.integral_dx += (dx + self.last_dx) * dt
        # self.integral_dy += (dy + self.last_dy) * dt
        leak = 1
        self.integral_dx = leak * self.integral_dx + (dx) * dt
        self.integral_dy = leak * self.integral_dy + (dy) * dt


        # Windup Clamp
        self.integral_dx = max(min(self.integral_dx, self.imax), -self.imax)
        self.integral_dy = max(min(self.integral_dy, self.imax), -self.imax)

        self.last_dx = dx
        self.last_dy = dy
        self.last_dz = dz

        vx = self.Kp * dx - self.Kd * self.dx_filtered + self.Ki * self.integral_dx
        vy = self.Kp * dy - self.Kd * self.dy_filtered + self.Ki * self.integral_dy

        vx = max(min(vx, 20), -20)
        vy = max(min(vy, 20), -20)

        # rospy.loginfo(
        # f"[Centering] dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f} | "
        # f"vx={vx:.3f}, vy={vy:.3f} | "
        # f"ddx={self.dx_filtered:.3f}, ddy={self.dy_filtered:.3f} | dt={dt:.4f}"
        # f"intx={self.integral_dx:.3f}, inty={self.integral_dy:.3f}"
        # )
        # # msg = Vector3()
        # msg.x = dx
        # msg.y = dy
        # msg.z = ddx 
        # self.debug_pub.publish(msg)
        self.move.linear.x = -vy
        self.move.linear.y = -vx
        self.move.linear.z = dz
        self.pub.publish(self.move)
        rgb_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Draw direction arrow (OpenCV only)
        rgb_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_img.shape
        cx, cy = w // 2, h // 2
        dx_prime = 100 * dx
        dy_prime = 100 * dy
        ex, ey = int(cx + dx_prime), int(cy + dy_prime)

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
        #cv2.imshow("Direction", image_with_vector)
        cv2.waitKey(1)
    
    def baby_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        if self.baby_is_stable is True:
            count = 0
            while (count < 10):
                words = process_image(cv_image)
                if words is None:
                    print("upper " + words[0] + " lower " + words[1])
                count += 1
        self.baby_is_stable = False
        


    def isMasterStable(self, dx, dy):
      if (dx**2 + dy**2) > 1:
        self.number_stable_frames = 0
        self.is_master_stable = False
        return
      self.number_stable_frames += 1
      if self.number_stable_frames > 100:
        self.is_master_stable = True
    
    def deleteTarget(self, board, target):
        tolerance = 0.005
        if self.baby.at_target is False or abs(self.baby.last_dz - consts.TARGET_HEIGHT) <= tolerance:
            self.stable_baby_frames = 0
            return
        self.stable_baby_frames += 1
        if self.stable_baby_frames < 100:
            return
        self.stable_baby_frames = 0
        self.baby_is_stable = True
        print("baby has stabilized")
        for element in self.boards[:]:
            if element[0] == board:
                if target in element[1]:
                    element[1].remove(target)
                    print("target removed")

                if not element[1]:
                    self.boards.remove(element)
                    print("board removed")

        

def main():
    rospy.init_node('robot_controller')
    try:
        mover = Mover()
    except Exception as e:
        rospy.logerr("Failed to initialize Mover: %s", e)
        return
    rospy.spin()

if __name__ == '__main__':
    main()
