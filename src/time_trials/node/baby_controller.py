import rospy
import numpy as np
import math
import cv2
from sensor_msgs.msg import LaserScan
from targeting import is_at_target
import constants

class BabyPID:
    def __init__(self):
        # PID gains
        self.Kp = 0.5
        self.Kd = 0.001
        self.Ki = 0.002
        self.imax = 25

        rospy.Subscriber('/Follower/rrbot/height', LaserScan, self.height_callback)

        # State variables
        self.at_target = False
        self.height = 0.0
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.last_dz = 0.0
        self.last_da = 0.0
        self.integral_dx = 0.0
        self.integral_dy = 0.0
        self.dx_filtered = 0.0
        self.dy_filtered = 0.0

        self.cruise_altitude = 0.5
        self.last_time = rospy.Time.now().to_sec()

    def height_callback(self, data):
       valid_ranges = [r for r in data.ranges if r != float('inf')]

       if len(valid_ranges) > 0:
          avg_height = sum(valid_ranges) / len(valid_ranges)
          self.height = avg_height
       else:
          self.height = 0.0

    def calculate_action(self, babyDrone, target, board, image):
        cx, cy, angle = babyDrone
        if target is None:
            return [0.0, 0.0, 0.0]
        tx, ty = target
        
        self.at_target = is_at_target(babyDrone, target)

        dz = 0
        if self.at_target == True:
            dz = constants.TARGET_HEIGHT - self.height
        else:
            cruise_altitude = self.calculateCruiseAltitude(babyDrone)
            dz = cruise_altitude - self.height

        # Time delta
        now = rospy.Time.now().to_sec()
        dt = now - self.last_time
        self.last_time = now
        if dt <= 0: dt = 1e-6

        # Positional error
        dx = tx - cx
        dy = ty - cy

        dx /= 100
        dy /= 100

        # Derivative
        ddx = (dx - self.last_dx) / dt
        ddy = (dy - self.last_dy) / dt
        alpha = 0.5
        self.dx_filtered = alpha * self.dx_filtered + (1-alpha) * ddx
        self.dy_filtered = alpha * self.dy_filtered + (1-alpha) * ddy

        # Integral with optional leak
        leak = 1.0
        self.integral_dx = leak * self.integral_dx + dx * dt
        self.integral_dy = leak * self.integral_dy + dy * dt

        # Clamp integrals
        self.integral_dx = max(min(self.integral_dx, self.imax), -self.imax)
        self.integral_dy = max(min(self.integral_dy, self.imax), -self.imax)

        # Save last error
        self.last_dx = dx
        self.last_dy = dy
        self.last_dz = dz

        # PID output
        vx = self.Kp * dx - self.Kd * self.dx_filtered + self.Ki * self.integral_dx
        vy = self.Kp * dy - self.Kd * self.dy_filtered + self.Ki * self.integral_dy

        world_vx = max(min(vx, 20), -20)
        world_vy = max(min(vy, 20), -20)

        # Heading correction (same as before)
        bx,by = board
        target_angle = math.atan2(by-cy, bx-cx)

        angle_error = angle - target_angle
        angle_error = (angle_error + math.pi) % (2*math.pi) - math.pi
        Ka = 1.0
        world_rot = angle_error * Ka

        self.last_da = angle_error

        Kz = 1.0
        drone_vz = Kz * dz
        drone_vx = world_vx * np.cos(angle) + world_vy * np.sin(angle)
        drone_vy = world_vx * np.sin(angle) - world_vy * np.cos(angle)
        
        #self.visualizeCommand(image, babyDrone, (drone_vx, drone_vy), target, target_angle)
        
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

    def calculateCruiseAltitude(self, baby_drone):
        x, y, _ = baby_drone
        cx = (constants.MAP_WIDTH/constants.SCALE_FACTOR)/2
        cy = (constants.MAP_HEIGHT/constants.SCALE_FACTOR)/2
        x_c = x - cx
        y_c = y - cy
        print("current pos x_c,y_c " + str(x_c) + ',' + str(y_c))
        if x < 600 or x > 990:
            print("cruise altitude of 0.5")
            return 0.5
        elif y > 400: 
            return 0.5
        else:
            print("cruse altitude of 2")
            return 2





def to_px(x, y, image):
    h, w = image.shape[:2]
    return int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))