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
from pid_controller import PIDController

class Mover:
    def __init__(self):
        self.master_controller = PIDController(pid_constants=(1.5, 0.3, 0.2), imax=10)
        self.baby = BabyPID()
        rospy.Subscriber('/Follower/rrbot/camera1/image_raw', Image, self.baby_camera_callback)
        rospy.Subscriber('/Master/rrbot/camera1/image_raw', Image, self.master_camera_callback)
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
        self.height = -1

        # System Stability States
        self.prev_baby_location = (0,0,0)
        self.is_master_stable = False
        self.number_stable_frames = 0
        self.stable_baby_frames = 0
        self.baby_is_stable = False
        self.first_time_stable = True
        rospy.sleep(1.0)

    def height_callback(self, data):
       valid_ranges = [r for r in data.ranges if r != float('inf')]

       if len(valid_ranges) > 0:
          avg_height = sum(valid_ranges) / len(valid_ranges)
          self.height = avg_height
       else:
          self.height = -1
          
    def master_camera_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # stabilize the master from its lookout point
        self.stabilize_master(cv_image)
        if self.is_master_stable is not True:
            self.move_baby.linear.x = 0
            self.move_baby.linear.y = 0
            self.move_baby.linear.z = 0
            self.move_baby.angular.z = 0
            self.baby_pub.publish(self.move_baby)
            return
        # If the master has just stabilized itself at its lookout, find the clue boards
        if self.first_time_stable is True:
            self.first_time_stable = False
            self.boards = find_clue_boards(cv_image)

        map = isolate_map(cv_image, int(consts.MAP_WIDTH/consts.SCALE_FACTOR), int(consts.MAP_HEIGHT/consts.SCALE_FACTOR))
        if consts.DEBUG is True:
            rgb_map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
            cv2.imshow("map", rgb_map)

        # Find the baby drone
        babyDrone = find_babyDrone(map, self.prev_baby_location)
        board, target = find_target(babyDrone, self.boards)
        if board is None:
            return
        self.process_target(board, target)

        # Find and go to the nearest target
        baby_vx, baby_vy, baby_angularz, baby_vz = self.baby.calculate_action(babyDrone, target, board, map)
        self.move_baby.linear.x = baby_vx
        self.move_baby.linear.y = baby_vy 
        self.move_baby.angular.z = baby_angularz
        self.move_baby.linear.z = baby_vz
        self.baby_pub.publish(self.move_baby)
        self.prev_baby_location = babyDrone

        
    def stabilize_master(self, cv_image):
        dx,dy,dz = find_course_center(cv_image)
        # prescaling
        dx /= 100
        dy /= 100

        vx, vy, vz = self.master_controller.PID((dx, dy, dz))
        self.check_master_stability(dy, dx)

        self.move.linear.x = -vy
        self.move.linear.y = -vx
        self.move.linear.z = dz
        self.pub.publish(self.move)

        if consts.DEBUG is True:
            debugging_visulization(cv_image, dx, dy)
        
    
    def baby_camera_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        if self.baby_is_stable is True:
            count = 0
<<<<<<< HEAD
            while (count < 1):
                upperWord, lowerWord = process_image(cv_image)
                if upperWord is not None and lowerWord is not None:
                    print("upper " + upperWord + " lower " + lowerWord)
=======
            # TODO read the words properly. The drone can be moving here
            while (count < 10):
                words = process_image(cv_image)
                if words is not None:
                    print("upper " + words[0] + " lower " + words[1])
>>>>>>> 46c72ba639ec26682042492edb0c35d929ccd970
                count += 1
        self.baby_is_stable = False
        


    def check_master_stability(self, dx, dy):
        # verify that the magnitude of the error function has remained below a threshold
        if (dx**2 + dy**2) > 1:
            self.number_stable_frames = 0
            self.is_master_stable = False
            return
        self.number_stable_frames += 1
        if self.number_stable_frames > 100:
            self.is_master_stable = True
    
    def process_target(self, board, target):
        tolerance = 0.25
        # TODO this logic is a bit brokey
        if self.baby.at_target is False or abs(self.baby.last_dz - consts.TARGET_HEIGHT) >= tolerance:
            # print("dz error " + str(self.baby.last_dz - consts.TARGET_HEIGHT))
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


def debugging_visulization(cv_image, dx, dy):
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
        cv2.imshow("Direction", image_with_vector)
        cv2.waitKey(1)     

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
