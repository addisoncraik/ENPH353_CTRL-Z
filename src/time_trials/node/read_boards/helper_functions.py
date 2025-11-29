import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

import constants as consts

def read_single_char_cnn(char_image):
  """
  Resizes, normalizes, and runs a prediction on a single cropped image.
  """

  if consts.CNN_MODEL is None:
      return '?' # Return placeholder if model failed to load

  # 1. Resize/Reshape to match training input (145x100x3)
  resized_image = cv2.resize(char_image, (consts.CHAR_WIDTH, consts.CHAR_HEIGHT))

  # 2. Normalize (0-255 -> 0.0-1.0)
  normalized_image = resized_image.astype('float32') / 255.0

  # 3. Add Batch Dimension (Keras expects (1, H, W, C) )
  input_tensor = np.expand_dims(normalized_image, axis=0)

  consts.CNN_MODEL.set_tensor(consts.CNN_INPUTS[0]['index'], input_tensor)
  consts.CNN_MODEL.invoke()

  # 4. Predict
  prediction_probs = consts.CNN_MODEL.get_tensor(consts.CNN_OUTPUTS[0]['index'])
  predicted_index = np.argmax(prediction_probs)

  # Standard mapping for model (A-Z)
  ALPHANUMERIC_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

  if predicted_index < len(ALPHANUMERIC_CHARS):
      return ALPHANUMERIC_CHARS[predicted_index]
  else:
      return '?'



def insert_spaces(word_list, avg_width):
  """
  Inserts a placeholder tuple ('space_x', 'space_y', 'space_w', 'space_h') into the list
  if the gap between two characters exceeds 0.7 times the average character width.
  """
  formatted_list = []

  for i, char in enumerate(word_list):

      # Skip space checking for the first character
      if i > 0:
          prev_char = word_list[i-1]

          # Gap distance: current character's X position - (previous character's X + previous character's Width)
          # Note: 'prev_char' is a tuple (x, y, w, h), so prev_char[0] is x and prev_char[2] is w
          gap_start_x = prev_char[0] + prev_char[2]
          gap_distance = char[0] - gap_start_x

          # Check for space (e.g., gap is larger than 1.0 * average character width)
          # We use 1.0 * avg_width as the threshold to distinguish a space from normal kerning.
          if gap_distance > (avg_width*0.7):
              # Insert a placeholder for a space
              formatted_list.append(("space_x", "space_y", "space_w", "space_h")) #arbitrary height

      # Add the current character
      formatted_list.append(char)

  return formatted_list


def four_point_transform(image, pts):
    """
    Warps a 4-point ROI into a flat rectangle.
    """

    maxWidth, maxHeight = consts.CLUEBOARD_WIDTH, consts.CLUEBOARD_HEIGHT

    # Source Points
    rect = np.array(pts, dtype="float32")

    # Destination points (Standard "Flat" View)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Calculate Homography and Warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped, M


def find_quadrant_corners(contour, approx_factor=0.01):
  """
  Finds the 4 'corner' points of the object by finding the furthest point
  from the centroid in each of the four quadrants.

  Args:
      contour: The largest contour found (assumed to be the board).

  Returns:
      np.array: A 4x2 array of the four corner points.
  """

  if contour is None or cv2.contourArea(contour) == 0:
      return None

  # 1. Calculate Perimeter
  peri = cv2.arcLength(contour, True)
  epsilon = approx_factor * peri

  # 2. Simplify Contour to Polygon (The result contains the "pointy parts")
  # The output shape is (N, 1, 2)
  polygon_vertices = cv2.approxPolyDP(contour, epsilon, True)

  # 3. Reshape and Return
  # Reshape from (N, 1, 2) to (N, 2) to get a clean array of points
  pts = polygon_vertices.squeeze()

  # Ensure the result is still a 2D array even if only one point is found
  if pts.ndim == 1:
      pts = np.expand_dims(pts, axis=0)

  min_x = min(pts[:, 0])
  max_x = max(pts[:, 0])
  min_y = min(pts[:, 1])
  max_y = max(pts[:, 1])

  cx = (min_x+max_x)/2
  cy = (min_y+max_y)/2


  # Initialize corners with the centroid and a tracking distance
  corners = np.full((4, 2), [cx, cy], dtype=np.float32)
  max_dist_sq = np.zeros(4, dtype=np.float32) # Stores squared distance for comparison

  # 3. Iterate through all contour points
  for pt in pts:
      x, y = pt
      dist_sq = (x - cx)**2 + (y - cy)**2

      if x-cx == 0:
        angle = math.pi
      else:
        angle = abs(np.arctan(abs(y - cy)/abs(x - cx)))

      # Determine quadrant index
      if x >= cx and y < cy:        # Top-Right (TR)
          q_idx = 1
      elif x < cx and y < cy:       # Top-Left (TL)
          q_idx = 0
      elif x < cx and y >= cy:      # Bottom-Left (BL)
          q_idx = 3
      else:                         # Bottom-Right (BR)
          q_idx = 2

      # Update corner if current point is further from centroid than the stored corner
      if dist_sq > max_dist_sq[q_idx] and angle > math.pi/30:
          max_dist_sq[q_idx] = dist_sq
          corners[q_idx] = pt

  return corners







# END OF FUNCTIONAL CODE









# Purely Visual, no function

def draw_ordered_corners(image, pts):
  """
  Draws the 4 ordered corner points (TL, TR, BR, BL) on the image
  with color-coded circles and labels for verification.

  Args:
      image (np.array): The source image.
      pts (np.array): 4 ordered points (TL, TR, BR, BL).

  Returns:
      np.array: Image with circles and text drawn.
  """
  # Create a copy to draw on
  debug_img = image.copy()

  # Define colors and labels based on the canonical order: [TL, TR, BR, BL]
  colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)] # Green, Red, Blue, Yellow (BGR)
  labels = ["TL", "TR", "BR", "BL"]

  # Iterate through the 4 points
  for i, point in enumerate(pts):
      (x, y) = point.ravel().astype(int)
      color = colors[i]
      label = labels[i]

      # Draw a large circle at the corner
      cv2.circle(debug_img, (x, y), 8, color, -1)

      # Add text label (e.g., "TL", "TR")
      cv2.putText(debug_img, label, (x + 10, y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

  return debug_img


def info(pts, characters, lower_word_images, upper_word_images, search_img, image_with_rects):
  # --- DISPLAY ---


  # Display Characters

  if len(characters) != 0:

    plt.figure(figsize=(18, 6))

    num_chars_lower = len(lower_word_images)
    if num_chars_lower > 0:
      for i, char_img in enumerate(lower_word_images):

        if not isinstance(char_img, np.ndarray):
          continue

        # i is the index (0, 1, 2, ...). We use i+1 for the subplot position (1, 2, 3, ...)
        ax = plt.subplot(1, num_chars_lower, i + 1)

        # Check if the item is a tuple placeholder (the space)
        if isinstance(char_img, tuple) and char_img[0] == 'NULL':
            # If it's the space placeholder, draw a simple empty box
            ax.set_title(f'[SPACE]')
            ax.text(0.5, 0.5, 'SPACE', fontsize=16, ha='center', va='center', color='gray')
            ax.axis('off')
        else:
            # Otherwise, draw the image
            ax.imshow(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
            ax.set_title(f'Character {i + 1}')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Plot 2: Upper Word Characters
    plt.figure(figsize=(18, 6))
    num_chars_upper = len(upper_word_images)
    if num_chars_upper > 0:
      for i, char_img in enumerate(upper_word_images):

        if not isinstance(char_img, np.ndarray):
          continue

        ax = plt.subplot(1, num_chars_upper, i + 1)

        # Check if the item is a tuple placeholder (the space)
        if isinstance(char_img, tuple) and char_img[0] == 'NULL':
            ax.set_title(f'[SPACE]')
            ax.text(0.5, 0.5, 'SPACE', fontsize=16, ha='center', va='center', color='gray')
            ax.axis('off')
        else:
            ax.imshow(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
            ax.set_title(f'Character {i + 1}')
            ax.axis('off')

    plt.tight_layout()
    plt.show()


  plt.figure(figsize=(18, 6))

  ax1 = plt.subplot(1, 2, 1)
  # Draw the found contour on the original image
  debug_img = search_img.copy()
  ax1.imshow(cv2.cvtColor(draw_ordered_corners(debug_img,pts), cv2.COLOR_BGR2RGB))
  ax1.set_title('1. Detected Board (4-Point Poly)')
  ax1.axis('off')

  ax3 = plt.subplot(1, 2, 2)
  ax3.imshow(cv2.cvtColor(image_with_rects, cv2.COLOR_BGR2RGB))
  ax3.set_title(f'3. Characters')
  ax3.axis('off')

  plt.tight_layout()
  plt.show()