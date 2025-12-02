import cv2
import numpy as np

import constants as consts

from .helper_functions import find_quadrant_corners
from .helper_functions import four_point_transform
from .helper_functions import insert_spaces
from .helper_functions import read_single_char_cnn
from .helper_functions import info

def process_image(search_img):
    if search_img is None: 
      print("No Image!")
      return None, None

    # --- STEP 1: COLOR FILTERING ---
    if consts.DEBUG:
      cv2.imshow("Search Image",search_img)
      print("\n--- STEP 1: Finding Blue Board Contour ---")

    # Determine Target Color
    target_color = np.array([120, 155, 0]) #Very Important
    tolerance = np.array([20, 100, 200]) # Don't change unless something broke

    lower_bound = np.clip(target_color - tolerance, [0, 0, 0], [179, 255, 255])
    upper_bound = np.clip(target_color + tolerance, [0, 0, 0], [179, 255, 255])

    image_hsv = cv2.cvtColor(search_img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise

    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        cv2.imshow("mask", mask)
        print("ERROR: No target regions found.")
        return None, None


    # Find the largest contour

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = sorted_contours[0]

    if consts.DEBUG:
      print("Found Clue Board")

    # Get the 4 corners from the contour
    pts = find_quadrant_corners(c)

    # --- STEP 2: PERSPECTIVE RECTIFICATION ---
    if consts.DEBUG:
      print("\n--- STEP 2: Rectifying Perspective ---")

    # unwarp the image using our pts
    unwarped_image, H_matrix = four_point_transform(search_img, pts)

    h, w = unwarped_image.shape[:2]

    if consts.DEBUG:
      cv2.imshow("Rectified Image",unwarped_image)
      print(f"Rectified Image Size: {w}x{h}")




    # --- STEP 3: CHARACTER DETECTION (HIERARCHY) ---
    if consts.DEBUG:
      print("\n--- STEP 3: Detecting Characters ---")

    BUFFER = 3 # Increases size of characters to capture more information
    MIN_AREA = 100 # Remove Noise


    target_color = np.array([120, 225, 175])
    tolerance = np.array([10, 35, 80])

    lower_bound = np.clip(target_color - tolerance, [0, 0, 0], [179, 255, 255])
    upper_bound = np.clip(target_color + tolerance, [0, 0, 0], [179, 255, 255])

    # Apply color filter
    image_hsv = cv2.cvtColor(unwarped_image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

    # Get image data
    image_with_rects = unwarped_image.copy()
    img_h, img_w = unwarped_image.shape[:2]
    img_area = img_w * img_h

    # Get contours and hierarchy to find the characters
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    characters = []

    if hierarchy is not None:

      hierarchy = hierarchy[0]

      if consts.DEBUG:
        print(f"Found {len(contours)} raw contours. Removing Noise and Filtering by shape...")
      #print(hierarchy)

      # Store border contours (inner & outer)
      border_indices = []
      areas = []

      # Identify the Outer Border (The Parent)
      for i, c in enumerate(contours):
          x, y, w_char, h_char = cv2.boundingRect(c)
          area = w_char*h_char

          if area < 0.001*img_area:
            continue

          if area > (0.10 * img_area):
            border_indices.append(i)
            continue

          areas.append(area)

      # Filter and Draw Characters Contours (The Children)
      for i, c in enumerate(contours):
          x, y, w_char, h_char = cv2.boundingRect(c)
          area = w_char*h_char

          # Filter 1: Ignore Noise and border
          if not (area in areas):
            continue

          # Filter 2: Check Parent Structure
          parent_idx = hierarchy[i][3]

          # Ensures no children of characters are counted
          if (parent_idx in border_indices) or (parent_idx == -1):
              characters.append((x-BUFFER, y-BUFFER, w_char+BUFFER*2, h_char+BUFFER*2))
              cv2.rectangle(image_with_rects, (x-BUFFER, y-BUFFER), (x + w_char+BUFFER, y + h_char+BUFFER), (0, 0, 255), 2)

    if consts.DEBUG:
      print(f"Total characters: {len(characters)}")
    # if consts.DEBUG:
    #   cv2.imshow("Step 3 - Character Mask", mask)
    #   cv2.imshow("Step 3 - Characters Detected", image_with_rects)



    # --- STEP 4: CHARACTER Organization ---
    if consts.DEBUG:
      print("\n--- STEP 4: Organizing Characters ---")

    if not characters:
        print("No characters found. Exiting.")
        return None, None
    else:
      y_pos = [(c[1]) for c in characters]
      y_avg = (max(y_pos)+min(y_pos))/2

      widths = [c[2] for c in characters]
      avg_char_width = sum(widths) / len(widths)

      # Sort into upper and lower words
      lower_word_chars = []
      upper_word_chars = []

      for char in characters:
        x, y, w, h = char

        if y > y_avg:
          lower_word_chars.append(char)
        else:
          upper_word_chars.append(char)

      # Sort characters based on 'x' coordinate (index 0) of the bounding box
      lower_word_chars.sort(key=lambda c: c[0])
      upper_word_chars.sort(key=lambda c: c[0])

      # Check for spaces
      lower_word_chars = insert_spaces(lower_word_chars, avg_char_width)
      upper_word_chars = insert_spaces(upper_word_chars, avg_char_width)

      # Get array of character images
      lower_word_images = []
      for char in lower_word_chars:

        if char[0] == 'space_x':
          lower_word_images.append("SPACE")
          continue

        x, y, w, h = char

        char_img = unwarped_image[y:y+h, x:x+w]
        lower_word_images.append(char_img)
        # if consts.DEBUG:
        #   cv2.imshow("Character Crop", char_img)


      upper_word_images = []
      for char in upper_word_chars:

        if char[0] == 'space_x':
          upper_word_images.append("SPACE")
          continue

        x, y, w, h = char
        char_img = unwarped_image[y:y+h, x:x+w]
        upper_word_images.append(char_img)

      if consts.DEBUG:
        print("Characters Organized")

    # --- STEP 5: CHARACTER READING ---
    if consts.DEBUG:
      print("\n--- STEP 5: Reading Characters ---")

    upperWord = ""
    lowerWord = ""


    if len(characters) != 0:
      for char_img in upper_word_images:
        if not isinstance(char_img, np.ndarray):
          upperWord += " "
          continue
        upperWord += read_single_char_cnn(cv2.cvtColor(char_img,cv2.COLOR_RGB2BGR))

      for char_img in lower_word_images:
        if not isinstance(char_img, np.ndarray):
          lowerWord += " "
          continue
        lowerWord += read_single_char_cnn(cv2.cvtColor(char_img,cv2.COLOR_RGB2BGR))

    if consts.DEBUG:
      print(f"Upper Word: {upperWord}")
      print(f"Lower Word: {lowerWord}")

    #info(pts, characters, lower_word_images, upper_word_images, search_img, image_with_rects)

    return upperWord, lowerWord