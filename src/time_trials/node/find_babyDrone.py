import cv2
import numpy as np
import math

import constants as consts

def find_babyDrone(map, last_pos):
    target_color = consts.DRONE_COLOR
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
        return last_pos

    M = cv2.moments(max_contour)

    if M['m00'] == 0:
        return last_pos
    # find center
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    tolerance = np.array([10, 10, 10])
    target_color = consts.HEAD_COLOR
    lower_bound = np.clip(target_color - tolerance, [0, 0, 0], [255, 255, 255])
    upper_bound = np.clip(target_color + tolerance, [0, 0, 0], [255, 255, 255])

    mask = cv2.inRange(map, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("PART TWO BABYYYY", mask)
    try:
        max_contour = max(contours, key=cv2.contourArea)
    except:
        return last_pos

    M = cv2.moments(max_contour)

    if M['m00'] == 0:
        return last_pos

    hx = int(M['m10'] / M['m00'])
    hy = int(M['m01'] / M['m00'])


    if consts.DEBUG:
        debug = map.copy()
        cv2.circle(debug, (cx, cy), 3, (0, 0, 255), -1)
        cv2.circle(debug, (hx, hy), 3, (255, 0, 0), -1)

        cv2.imshow("baby_find", debug)

    angle = math.atan2(hy - cy, hx - cx)

    if consts.DEBUG:
        print(cx,cy,angle*180/math.pi)

    if consts.DEBUG:
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