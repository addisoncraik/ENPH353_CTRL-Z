import cv2
import numpy as np
import math

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
    return cv2.warpPerspective(camera_feed, M, (map_width, map_height)), M
