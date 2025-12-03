import cv2
import numpy as np
import matplotlib.pyplot as plt

def filter_for_course(image):
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Define HSV ranges for grey (from previous analysis)
  lower_grey_h1 = np.array([0, 0, 150])
  upper_grey_h1 = np.array([15, 50, 200])
  lower_grey_h2 = np.array([165, 0, 150])
  upper_grey_h2 = np.array([180, 50, 200])

  # Create masks for each hue range
  mask_h1 = cv2.inRange(hsv, lower_grey_h1, upper_grey_h1)
  mask_h2 = cv2.inRange(hsv, lower_grey_h2, upper_grey_h2)

  # Combine the two hue masks
  combined_mask = cv2.bitwise_or(mask_h1, mask_h2)

  return combined_mask

def process_image_for_centering(image):
  # Get the mask where the grey surface is white (255) and background is black (0)
  grey_mask = filter_for_course(image)

  # Invert the mask: grey surface becomes black (0), background becomes white (255)
  # This is for visualization where the 'course' is white and the 'grey surface' is black.
  inverted_mask = cv2.bitwise_not(grey_mask)

  # Count white pixels (representing the course)
  white_pixels = np.count_nonzero(inverted_mask == 255)
  # Count black pixels (representing the grey surface)
  black_pixels = np.count_nonzero(inverted_mask == 0)

  return inverted_mask, white_pixels, black_pixels

def find_course_center(image):
  # Call the updated process_image_for_centering to get pixel counts
  inverted_mask, white_pixels, black_pixels = process_image_for_centering(image)

  height, width = inverted_mask.shape[:2]
  x, y = width // 2, height // 2
  cX, cY = 0, 0
  M = cv2.moments(inverted_mask)
  if M["m00"] != 0:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
  else:
    cX, cY = x, y

  dx = cX - x
  dy = cY - y

  # Calculate dz for vertical adjustment based on pixel ratio
  dz = 0 # Default to no vertical movement
  total_pixels = white_pixels + black_pixels
  if total_pixels > 0:
    ratio_white_to_total = white_pixels / total_pixels

    # if (ratio_white_to_total < 0.40):
    #   dz = 0
    # else:
    dz = 10 * (ratio_white_to_total - 0.4)
    # Define thresholds for vertical adjustment
    # These thresholds might need tuning based on actual drone height and field of view
    too_low_threshold = 0.70 # If course takes up > 90% of frame, drone is too low
    too_high_threshold = 0.60 # If course takes up < 60% of frame (showing too much black background), drone is too high
    #dz should be < 0 if ratio is small. it should be big (up) when ratio is high
    
    # if ratio_white_to_total > too_low_threshold: # Course is occupying too much of the image (drone is too low)
    #   dz = 1 # Move up
    # elif ratio_white_to_total < too_high_threshold: # Course is occupying too little of the image (drone is too high)
    #   dz = -1 # Move down
    # else:
    #   dz = 0 # Optimal height

  return dx, dy, dz