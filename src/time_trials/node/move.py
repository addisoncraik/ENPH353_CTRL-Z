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
import math

MAP_WIDTH = 2000
MAP_HEIGHT = 900
SCALE_FACTOR = 2

DRONE_COLOR = np.array([17, 225, 255])
HEAD_COLOR = np.array([255, 255, 0])
DEBUG = True

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
        rospy.sleep(1.0)


    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        self.stabilize_master(cv_image)
        map = isolate_map(cv_image, int(MAP_WIDTH/SCALE_FACTOR), int(MAP_HEIGHT/SCALE_FACTOR))
        rgb_map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
        cv2.imshow("map", rgb_map)
        baby_x, baby_y, baby_theta = self.find_babyDrone(map)
        board, target = self.find_target((baby_x, baby_y, baby_theta), map)
        baby_vx, baby_vy, baby_angularz = self.calculate_action((baby_x, baby_y, baby_theta), target, board, map)
        self.move_baby.linear.x = -baby_vy
        self.move_baby.linear.y = -baby_vx
        self.move_baby.angular.z = baby_angularz
        self.baby_pub.publish(self.move_baby)




    @staticmethod
    def find_babyDrone(map):

        target_color = DRONE_COLOR
        tolerance = np.array([10, 20, 20])
        hsl_map = cv2.cvtColor(map, cv2.COLOR_RGB2HSV)
        
        lower_bound = np.clip(target_color - tolerance, [0, 0, 0], [179, 255, 255])
        upper_bound = np.clip(target_color + tolerance, [0, 0, 0], [179, 255, 255])

        msk = cv2.inRange(hsl_map, lower_bound, upper_bound)

        contours, _ = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("ballin", msk)
        try:
          max_contour = max(contours, key=cv2.contourArea)
        except:
          return (0, 0, 0)

        M = cv2.moments(max_contour)

        if M['m00'] == 0:
          return (0, 0, 0)
        # find center
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        tolerance = np.array([10, 10, 10])
        target_color = HEAD_COLOR
        lower_bound = np.clip(target_color - tolerance, [0, 0, 0], [255, 255, 255])
        upper_bound = np.clip(target_color + tolerance, [0, 0, 0], [255, 255, 255])

        mask = cv2.inRange(map, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("PART TWO BABYYYY", mask)
        try:
          max_contour = max(contours, key=cv2.contourArea)
        except:
          return (0, 0, 0)

        M = cv2.moments(max_contour)

        if M['m00'] == 0:
          return (0, 0, 0)

        hx = int(M['m10'] / M['m00'])
        hy = int(M['m01'] / M['m00'])


        if DEBUG:
          debug = map.copy()
          cv2.circle(debug, (cx, cy), 3, (0, 0, 255), -1)
          cv2.circle(debug, (hx, hy), 3, (255, 0, 0), -1)

          cv2.imshow("baby_find", debug)

        angle = math.atan2(cy - hy, cx - hx)

        if DEBUG:
          print(cx,cy,angle*180/math.pi)

        if DEBUG:
            debug = map.copy()

            # Draw center (red)
            cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)

            # Draw head (blue)
            cv2.circle(debug, (hx, hy), 5, (255, 0, 0), -1)

            # Draw orientation vector (green)
            length = 50  # length of orientation arrow
            ex = int(cx + length * math.cos(angle))
            ey = int(cy + length * math.sin(angle))
            cv2.arrowedLine(debug, (cx, cy), (ex, ey), (0, 255, 0), 3, cv2.LINE_AA)

            # Draw angle text
            angle_deg = angle * 180 / math.pi
            cv2.putText(debug, f"{angle_deg:.1f} deg",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            cv2.imshow("baby orientation", debug)

        return cx, cy, angle

        
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

    def find_target(self, babyDrone, camera_feed):
        cx, cy, _ = babyDrone

        if self.boards == []:
          return None, None

        cx *= SCALE_FACTOR
        cy *= SCALE_FACTOR

        distances = []

        for board in self.boards:
            bx, by = board[0]

            distances.append(math.sqrt((cx - bx) ** 2 + (cy - by) ** 2))

        closest_board = self.boards[distances.index(min(distances))]


        distances = []
        targets = closest_board[1]
        closest_board_index = self.boards.index(closest_board)


        for target in targets:
            tx, ty = target

            distance = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)

            if distance < 20/SCALE_FACTOR:
              self.boards[closest_board_index][1].remove(target)

            distances.append(distance)

        if len(targets) == 0:
          self.boards.remove(closest_board)
          return self.find_target(babyDrone, camera_feed)

        closest_target_index = distances.index(min(distances))

        if closest_target_index >= len(targets):
          closest_target_index = 0

        closest_target = targets[closest_target_index]


        return (closest_board[0][0]/SCALE_FACTOR, closest_board[0][1]/SCALE_FACTOR), (closest_target[0]/SCALE_FACTOR, closest_target[1]/SCALE_FACTOR)

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



def isolate_map(camera_feed, map_width, map_height):

    camera_feed_hsv = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2HSV)

    target_color = np.array([0, 0, 180])
    tolerance = np.array([300, 0, 10])

    lower_bound = np.clip(target_color - tolerance, [0, 0, 0], [179, 255, 255])
    upper_bound = np.clip(target_color + tolerance, [0, 0, 0], [179, 255, 255])


    msk = cv2.inRange(camera_feed_hsv, lower_bound, upper_bound)

    # invert mask
    msk_inv = cv2.bitwise_not(msk)

    contours, _ = cv2.findContours(msk_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)

    polygon_approximation = cv2.approxPolyDP(max_contour, 0.01 * cv2.arcLength(max_contour, True), True)
    pts = polygon_approximation.reshape(-1, 2)
    # pts = np.array(polygon_approximation).squeeze()
    src = np.zeros((4, 2), dtype="float32")
    center_point = np.zeros((2), dtype="float32")

    for pt in pts:
      center_point[0] += pt[0]
      center_point[1] += pt[1]

    center_point /= 4

    angles = []

    for pt in pts:
      angle = np.arctan2(pt[1]-center_point[1], pt[0]-center_point[0])

      if angle > 0 and angle < math.pi/2:
        src[2] = pt #bottom right
      elif angle > math.pi/2:
        src[3] = pt #bottom left
      elif angle > -math.pi/2 and angle < 0:
        src[1] = pt #top right
      else:
        src[0] = pt #top left

    dest = np.array([
        [0, 0],
        [map_width, 0],
        [map_width, map_height],
        [0, map_height]
    ], dtype=np.float32)

    # make transformation matrix
    M = cv2.getPerspectiveTransform(src, dest)
    return cv2.warpPerspective(camera_feed, M, (map_width, map_height))


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
