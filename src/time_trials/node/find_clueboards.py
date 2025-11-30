import cv2
import numpy as np

import constants as consts

from isolate_map import isolate_map

def find_clue_boards(img):
  if img is None:
    print(f"Error: Could not load image.")
    return

  print("Image loaded. Applying filters...")

  # --- 1. Isolate Map ---
  # Make large map so we can pick out all the time details
  map_img = isolate_map(img, consts.MAP_WIDTH, consts.MAP_HEIGHT)


  # --- 2. Define Target Color and Tolerance ---
  # Using your values for the blue poles
  target_color = np.array([119, 73, 210])
  tolerance = np.array([15, 5, 45])

  # --- 3. Calculate Filter Bounds ---
  lower_bound = np.clip(target_color - tolerance, [0, 0, 0], [179, 255, 255])
  upper_bound = np.clip(target_color + tolerance, [0, 0, 0], [179, 255, 255])


  # --- 4. Apply the Color Filter ---
  # Using HSV as in your original code
  map_hsv = cv2.cvtColor(map_img, cv2.COLOR_RGB2HSV)

  # Create the original mask (White = Found)
  mask = cv2.inRange(map_hsv, lower_bound, upper_bound)

  # Try to make mask more readable (May need to do more than just this)
  kernel = np.ones((7,7), np.uint8)
  solid_mask = cv2.dilate(mask, kernel, iterations=2)


  # Find the outlines (contours) of all the white areas in the mask
  contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  print(f"Found {len(contours)} raw contours. Removing Noise and Filtering by shape...")

  cv2.imshow("Clueboards Found", mask)
  all_boards = [] # This will store the (x,y) of all circles from detected boards

  # Loop over every contour found
  for contour in contours:
      area = cv2.contourArea(contour)

      _, _, w, h = cv2.boundingRect(contour)

      # find center
      M = cv2.moments(contour)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])

      # Assume hroizontal board
      target_1 = (cx/consts.SCALE_FACTOR, cy/consts.SCALE_FACTOR - consts.TARGET_OFFSET) #top
      target_2 = (cx/consts.SCALE_FACTOR, cy/consts.SCALE_FACTOR + consts.TARGET_OFFSET) #bottom

      #If it's vertical make proper adjustments
      if(float(w/h) < 1):
          target_1 = (cx/consts.SCALE_FACTOR - consts.TARGET_OFFSET, cy/consts.SCALE_FACTOR) #left
          target_2 = (cx/consts.SCALE_FACTOR + consts.TARGET_OFFSET, cy/consts.SCALE_FACTOR) #right

      #Save the target's coordinates

      board = [
          (cx/consts.SCALE_FACTOR,cy/consts.SCALE_FACTOR),
          [
              target_1,
              target_2
          ]
      ]

      all_boards.append(board)
  return all_boards