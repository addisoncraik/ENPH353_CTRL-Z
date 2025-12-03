#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu

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
from targeting import find_target

from baby_controller import BabyPID

from read_boards.process_image import process_image

import constants as consts
from pid_controller import PIDController

class Mover:
    def __init__(self):
        
        #Create Publishers
        self.debug_pub = rospy.Publisher("/centering_debug", Vector3, queue_size=1)
        self.pub = rospy.Publisher('/Master/cmd_vel', Twist, queue_size=1)
        self.baby_pub = rospy.Publisher('/Follower/cmd_vel', Twist, queue_size=1)
        self.score_tracker = rospy.Publisher('/score_tracker', String, queue_size=1)

        #Initialize Sub-Objects
        self.baby = BabyPID()
        self.bridge = CvBridge()
        self.move = Twist()
        self.move_baby = Twist()
        self.master_controller = PIDController(pid_constants=(1.5, 0.3, 0.2), imax=10)

        #Store Clueboard Locations
        self.boards = [
            [(500,250), [(490, 250), (510, 250)]]
            ]
        
        #Track Height
        self.height = -1

        #Track Whether we Have Succesfuly Read a Sign
        self.read = False

        #Track Whether we have finished the course
        self.sent_terminator = False

        # System Stability States
        self.prev_baby_location = (0,0,0)
        self.is_master_stable = False
        self.number_stable_frames = 0
        self.stable_baby_frames = 0
        self.baby_is_stable = False
        self.first_time_stable = True


        self.height_map = None
        self.transformed_height_map = None
        rospy.sleep(5.0)

    def height_callback(self, data):
       valid_ranges = [r for r in data.ranges if r != float('inf')]

       if len(valid_ranges) > 0:
          avg_height = sum(valid_ranges) / len(valid_ranges)
          self.height = avg_height
       else:
          self.height = -1

    def master_depth_callback(self, data):
        if self.is_master_stable:
            return
        
        try:
            raw_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return 

        
        self.height_map = np.clip(raw_image, 0.0, 7.5)

        print(
            "DEPTH:", 
            self.height_map.dtype, 
            "min:", np.min(self.height_map), 
            "max:", np.max(self.height_map), 
            "unique:", np.unique(self.height_map)[:10]
        )
          
    def master_camera_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
        if not self.boards and self.is_master_stable is True and self.sent_terminator is False:
            self.sent_terminator = True
            self.score_tracker.publish(str(consts.TEAM_ID+','+consts.TEAM_PASSWORD+',-1,xxxx'))

        # stabilize the master from its lookout point
        self.stabilize_master(cv_image)
        if self.is_master_stable is not True:
            return
        # If the master has just stabilized itself at its lookout, find the clue boards
        if self.first_time_stable is True:
            self.first_time_stable = False
            self.boards, H = find_clue_boards(cv_image)
            # self.transformed_height_map =cv2.warpPerspective(self.height_map, H, (consts.MAP_WIDTH,consts.MAP_HEIGHT))
            # Warp
            self.transformed_height_map = cv2.warpPerspective(
                self.height_map, H,
                (consts.MAP_WIDTH, consts.MAP_HEIGHT)
            )

            self.transformed_height_map = self.transformed_height_map.astype(np.float64)

            # --- Display warped image ---
            disp = cv2.normalize(self.transformed_height_map, None, 0, 1, cv2.NORM_MINMAX)
            cv2.imshow("trans 1", disp)
            cv2.waitKey(1)

            # Laplacian
            lap = cv2.Laplacian(self.transformed_height_map, cv2.CV_64F)

            # Display Laplacian nicely
            disp_lap = cv2.normalize(np.abs(lap), None, 0, 1, cv2.NORM_MINMAX)
            cv2.imshow("laplace", disp_lap)
            cv2.waitKey(1)

            # Gaussian blur
            blur = cv2.GaussianBlur(lap, (201,201), 0)

            # Display blur result
            disp_blur = cv2.normalize(np.abs(blur), None, 0, 1, cv2.NORM_MINMAX)
            cv2.imshow("blur + laplace", disp_blur)
            cv2.waitKey(1)

            self.transformed_height_map = cv2.resize(blur, (int(consts.MAP_WIDTH/consts.SCALE_FACTOR), int(consts.MAP_HEIGHT/consts.SCALE_FACTOR)))
            self.transformed_height_map = cv2.normalize(self.transformed_height_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)



        map, _ = isolate_map(cv_image, int(consts.MAP_WIDTH/consts.SCALE_FACTOR), int(consts.MAP_HEIGHT/consts.SCALE_FACTOR))
        if consts.DEBUG is True:
            rgb_map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
            # cv2.imshow("map", rgb_map)

        # Find the baby drone
        babyDrone = find_babyDrone(map, self.prev_baby_location)
        board, target = find_target(babyDrone, self.boards)
        if board is None:
            return
        self.process_target(board, target)

        # Find and go to the nearest target
        self.baby.calculate_action(babyDrone, target, board, map, self.transformed_height_map)
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
            
            finished_reading = False
            count = 0

            while (not finished_reading and count < 10):
                upperWord, lowerWord = process_image(cv_image)
                # TODO this is not what its reading
                #upperWord, lowerWord = "rain", "coat"
                if upperWord is not None and lowerWord is not None:

                    if upperWord in consts.DICTIONARY:
                        self.score_tracker.publish(str(consts.TEAM_ID+","+consts.TEAM_PASSWORD+","+consts.DICTIONARY[upperWord]+","+lowerWord))

                        self.read = True
                        finished_reading = True

                    if consts.DEBUG:
                        print("upper " + upperWord + " lower " + lowerWord)

                else: 
                    finished_reading = True

                count += 1
            
        self.baby_is_stable = False
        
    def check_master_stability(self, dx, dy):
        # verify that the magnitude of the error function has remained below a threshold
        if (dx**2 + dy**2) > 10:
            self.number_stable_frames = 0
            self.is_master_stable = False
            return
        self.number_stable_frames += 1
        if self.number_stable_frames > 100:
            self.is_master_stable = True
    
    def process_target(self, board, target):
        tolerance = 0.25
        if self.read is True:
            for element in self.boards:
                if element[0] == board and (len(element[1]) == 1 or len(element[1]) == 0):
                    self.boards.remove(element)
            self.read = False
            print("Clue Submitted, Board removed")
        # TODO this logic is a bit brokey
        if self.baby.at_target is False or abs(self.baby.last_dz - consts.TARGET_HEIGHT) >= tolerance or self.baby.aligned_with_target is False:
            #print("dz error " + str(self.baby.last_dz - consts.TARGET_HEIGHT))
            self.stable_baby_frames = 0
            return

        self.stable_baby_frames += 1
        if self.stable_baby_frames < 25:
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

                break


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
        #cv2.imshow("Direction", image_with_vector)
        # cv2.waitKey(1)     

def main():
    rospy.init_node('robot_controller')
    try:
        mover = Mover()
        #Subscribe to Topics
        rospy.Subscriber('/Follower/rrbot/camera1/image_raw', Image, mover.baby_camera_callback)
        rospy.Subscriber('/Master/rrbot/camera1/image_raw', Image, mover.master_camera_callback)
        rospy.Subscriber('/Master/rrbot/height', LaserScan, mover.height_callback)
        rospy.Subscriber('/Master/rrbot/camera1/depth/image_raw', Image, mover.master_depth_callback)
        #Start ScoreBoard
        mover.score_tracker.publish(str(consts.TEAM_ID+"," + consts.TEAM_PASSWORD+",0,xxxx"))

    except Exception as e:
        rospy.logerr("Failed to initialize Mover: %s", e)
        return
    rospy.spin()

if __name__ == '__main__':
    main()
