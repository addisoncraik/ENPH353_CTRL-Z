import rospy
import numpy as np
import math
import cv2
import tf
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Imu
from targeting import is_at_target
import constants
from pid_controller import PIDController
from geometry_msgs.msg import Twist


class BabyPID:
    def __init__(self):
        self.baby_controller = PIDController(pid_constants=(1.0, 0.1, 0.002), imax=25)
        self.roll_pitch_controller = PIDController(pid_constants=(10.0, 1.0, 0.00), imax=10.0, tuning=False)
        rospy.Subscriber('/Follower/rrbot/height', LaserScan, self.height_callback)
        rospy.Subscriber('/Master/rrbot/height', LaserScan, self.master_height_callback)
        rospy.Subscriber('/Follower/imu', Imu, self.imu_callback)


        self.baby_pub = rospy.Publisher('/Follower/cmd_vel', Twist, queue_size=1)
        self.translational_cmd = Twist()
        self.angular_cmd = Twist()
        self.overall_command = Twist()

        # State variables
        self.at_target = False
        self.aligned_with_target = False
        self.angular_stability = True
        self.angular_stability_counter = 0
        self.height = 0.0
        self.master_height = 0.0
        self.last_dz = 0.0
        self.wx = 0
        self.wy = 0
        self.cruise_altitude = 0.5

    def imu_callback(self, msg):
        orientation = msg.orientation
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
            # Desired orientation = level (no roll, no pitch)
        q_des = np.array([0.0, 0.0, 0.0, 1.0])

        # Quaternion error
        q_err = tf4.quaternion_multiply(q_des, tf4.quaternion_conjugate(q))

        # Small angle error vector
        angle_error = 2.0 * np.array(q_err[0:3])
        roll = angle_error[0]
        pitch = angle_error[1]
        if (roll**2 + pitch**2) > 0.1:
            self.angular_stability_counter = 20
            self.angular_stability = False
        elif self.angular_stability_counter > 0:
            self.angular_stability_counter -= 1
        else:
            self.angular_stability = True

        # PID
        roll_cmd, pitch_cmd, _ = self.roll_pitch_controller.PID((roll, pitch, 0))

        # Apply to actuators CORRECTLY
        self.wx = -roll_cmd     # roll actuator
        self.wy = -pitch_cmd    # pitch actuator
        # print("roll: " + str(roll) + " pitch: " + str(pitch))
        # print("response: roll: " + str(roll_cmd) + " pitc: " + str(pitch_cmd))
        self.angular_cmd.angular.x = self.wx
        self.angular_cmd.angular.y = self.wy
        self.publish()


    def publish(self):
        self.overall_command.angular = self.angular_cmd.angular
        if self.angular_stability is True:
            self.overall_command.linear = self.translational_cmd.linear
            self.overall_command.angular.z = self.translational_cmd.angular.z
        else:
            self.overall_command.linear.x = 0
            self.overall_command.linear.y = 0
            self.overall_command.linear.z = 0
            self.overall_command.angular.z = 0

        self.baby_pub.publish(self.overall_command)


    def height_callback(self, data):
       valid_ranges = [r for r in data.ranges if r != float('inf')]

       if len(valid_ranges) > 0:
          avg_height = sum(valid_ranges) / len(valid_ranges)
          self.height = avg_height
       else:
          self.height = 0.0

    def master_height_callback(self, data):
        valid_ranges = [r for r in data.ranges if r != float('inf')]
        if len(valid_ranges) > 0:
            avg_height = sum(valid_ranges) / len(valid_ranges)
            self.master_height = avg_height
        else:
            self.master_height = 0.0


    def calculate_action(self, babyDrone, target, board, image, height_map=None):
        cx, cy, angle = babyDrone
        if target is None:
            return [0.0, 0.0, 0.0]
        tx, ty = target
        cx_centered, cy_centered = center_points(cx, cy)
        tx_centered, ty_centered = center_points(tx, ty)

        # adjust for parrallel axis
        delta_x = 0
        delta_y = 0
        if self.master_height != 0.0 and cx_centered > 0:
            delta_x = self.height / self.master_height * cx_centered
            delta_y = self.height / self.master_height * cy_centered 
        
        cx_mod = cx_centered - delta_x
        cy_mod = cy_centered - delta_y

        # Positional error
        dx = tx_centered - cx_mod
        dy = ty_centered - cy_mod

        dx /= 100
        dy /= 100
        
        dz = 0
        self.at_target = is_at_target((cx_mod, cy_mod, angle), (tx_centered, ty_centered))
        height_scaling = 0
        if height_map is not None:
            height_scaling = height_map[cy, cx]
        if self.at_target == True:
            dz = constants.TARGET_HEIGHT - self.height
        else:
            # cruise_altitude = self.calculateCruiseAltitude(babyDrone)
            dz = constants.CRUISE_ALTITUDE + 4*height_scaling - self.height
        self.last_dz = dz
        # Heading correction (same as before)
        bx, by = board
        target_angle = math.atan2(by - cy, bx - cx)

        angle_error = angle - target_angle
        angle_error = (angle_error + math.pi) % (2*math.pi) - math.pi

        if abs(angle_error) > 0.1:
            self.aligned_with_target = False
        else:
            self.aligned_with_target = True
        
        world_vx, world_vy, world_rot = self.baby_controller.PID((dx, dy, angle_error))

        Kz = 2.0
        drone_vz = Kz * dz
        drone_vx = world_vx * np.cos(angle) + world_vy * np.sin(angle)
        drone_vy = world_vx * np.sin(angle) - world_vy * np.cos(angle)
        
        self.visualizeCommand(image, babyDrone, (drone_vx, drone_vy), target, target_angle)

        self.translational_cmd.linear.x = drone_vx
        self.translational_cmd.linear.y = drone_vy
        self.translational_cmd.linear.z = drone_vz
        self.translational_cmd.angular.z = world_rot
        self.publish()
        return [drone_vx, drone_vy, world_rot, drone_vz]

    def visualizeCommand(self, image, baby_pos, baby_cmd, target, target_angle):
        drone_vx, drone_vy = baby_cmd
        temp = drone_vx
        drone_vx = -drone_vy
        drone_vy = -temp
        cx, cy, angle = baby_pos
        tx, ty = target
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


    # TODO this is so cooked hopefully camera inverse transform eliminates this
    def calculateCruiseAltitude(self, baby_drone):
        x, y, _ = baby_drone
        cx = (constants.MAP_WIDTH/constants.SCALE_FACTOR)/2
        cy = (constants.MAP_HEIGHT/constants.SCALE_FACTOR)/2
        x_c = x - cx
        y_c = y - cy
        # print("current pos x_c,y_c " + str(x_c) + ',' + str(y_c))
        if x < 600 or x > 990:
            # print("cruise altitude of 0.5")
            return 0.5
        elif y > 400: 
            return 0.5
        # elif x > 600 and x < 700:
        #     return 4
        else:
            # print("cruse altitude of 2")
            return 2

def center_points(x,y):
    cx = (constants.MAP_WIDTH/constants.SCALE_FACTOR)/2
    cy = (constants.MAP_HEIGHT/constants.SCALE_FACTOR)/2
    x_c = x - cx
    y_c = y - cy

    return x_c, y_c


def to_px(x, y, image):
    h, w = image.shape[:2]
    return int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
