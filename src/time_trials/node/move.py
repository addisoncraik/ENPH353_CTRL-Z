#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from isolate_map import isolate_map
from find_center import find_course_center
from find_clueboards import find_clue_boards

from find_babyDrone import find_babyDrone
from targeting import is_at_target
from targeting import find_target

from read_boards.process_image import process_image

import constants as consts

class Mover:
    def __init__(self):
        rospy.Subscriber('/Master/rrbot/camera1/image_raw', Image, self.callback)
        self.pub = rospy.Publisher('/Master/cmd_vel', Twist, queue_size=1)
        self.baby_pub = rospy.Publisher('/Follower/cmd_vel', Twist, queue_size=1)
        self.score_tracker = rospy.Publisher('/score_tracker', String, queue_size=1)
        self.bridge = CvBridge()
        self.move = Twist()
        self.move_baby = Twist()
        self.boards = [[(1000,450), [(1010, 450), (990, 450)]]]
        self.last_dx = 0
        self.last_dy = 0
        self.last_time = rospy.Time.now().to_sec()

        self.last_baby_pos = (0,0,0)
        rospy.sleep(1.0)


    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        self.stabilize_master(cv_image)
        map = isolate_map(cv_image, int(consts.MAP_WIDTH/consts.SCALE_FACTOR), int(consts.MAP_HEIGHT/consts.SCALE_FACTOR))
        rgb_map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
        cv2.imshow("map", rgb_map)
        babyDrone = find_babyDrone(map, self.last_baby_pos)
        board, target = find_target(babyDrone, self.boards)
        baby_vx, baby_vy, baby_angularz = self.calculate_action(babyDrone, target, board, map)
        self.move_baby.linear.x = -baby_vy
        self.move_baby.linear.y = -baby_vx
        self.move_baby.angular.z = baby_angularz
        self.baby_pub.publish(self.move_baby)

        self.last_baby_pos = babyDrone

        
    def stabilize_master(self, cv_image):

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

        Kp = 0.01  # proportional gain
        Kd = 0.001    # damping gain

        vx = Kp * dx - Kd * ddx
        vy = Kp * dy - Kd * ddy

        # Optional clamp
        vx = max(min(vx, 5), -5)
        vy = max(min(vy, 5), -5)

        # rospy.loginfo(
        # f"[Centering] dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f} | "
        # f"vx={vx:.3f}, vy={vy:.3f} | "
        # f"ddx={ddx:.3f}, ddy={ddy:.3f} | dt={dt:.4f}"
        # )

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

    ## TODO: Brokey
    def calculate_action(self, babyDrone, target, board, image):
        cx, cy, angle = babyDrone
        if target is None:
            return [0, 0, 0]

        tx, ty = target
        bx, by = board

        action = [0.0, 0.0, 0.0]

        # Find the positional error
        dx = tx - cx
        dy = ty - cy

        Kp = 0.01
        world_vx = dx * Kp
        world_vy = dy * Kp

        ## Calculate angle to target
        target_angle = math.atan2(dy, dx)
        angle_error = target_angle - angle
        angle_error = (angle_error + math.pi) % (2*math.pi) - math.pi
        Ka = 0.5
        world_rot = angle_error * Ka

        drone_vx = world_vx * np.cos((angle - np.pi/2)) - world_vy * np.sin((angle - np.pi/2))
        drone_vy = world_vx * np.sin((angle - np.pi/2)) + world_vy * np.cos((angle - np.pi/2))
        # drone_vx = world_vx
        # drone_vy = world_vy

        image_with_vector = image.copy()
        px1, py1 = to_px(cx, cy, image)
        px2, py2 = to_px(tx, ty, image)

        cv2.arrowedLine(
            image_with_vector,
            (px1, py1),
            (px2, py2),
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )
        # --- Draw commanded motion vector (RED) ---

        scale = 500  # increase length so it's visible
        cmd_end_x = int(cx + drone_vx * scale)
        cmd_end_y = int(cy + drone_vy * scale)

        px_cmd1, py_cmd1 = to_px(cx, cy, image)
        px_cmd2, py_cmd2 = to_px(cmd_end_x, cmd_end_y, image)

        cv2.arrowedLine(
            image_with_vector,
            (px_cmd1, py_cmd1),
            (px_cmd2, py_cmd2),
            (0, 0, 255),   # red
            3,
            cv2.LINE_AA
        )
                # orientation vector (blue)
        hx = int(cx + 50 * math.cos(angle))
        hy = int(cy + 50 * math.sin(angle))
        cv2.arrowedLine(image_with_vector, (cx, cy), (hx, hy), (255, 0, 0), 2)

        # desired direction vector (green)
        txv = int(cx + 50 * math.cos(target_angle))
        tyv = int(cy + 50 * math.sin(target_angle))
        cv2.arrowedLine(image_with_vector, (cx, cy), (txv, tyv), (0, 255, 0), 2)

        cv2.putText(image_with_vector, "cmd",
                    (px_cmd2 + 5, py_cmd2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

        # Display live with OpenCV (non-blocking)
        cv2.imshow("Baby Controller", image_with_vector)
        cv2.waitKey(1)

        action[0] = drone_vx  # Forward/Back
        action[1] = drone_vy  # Left/Right
        action[2] = world_rot  # Rotation

        return action
# safe converter for world→image  
def to_px(x, y, image):
    h, w = image.shape[:2]
    return int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))


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
