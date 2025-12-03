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
  tolerance = np.array([15, 10, 50])

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

  debug = mask.copy()

  filtered_contours = []

  for c1 in contours:
    x1,y1,w1,h1 = cv2.boundingRect(c1)

    cv2.rectangle(debug,(x1,y1), (x1+w1,y1+h1), (255,0,0), 2)

    cx1 = x1+w1/2
    cy1 = y1+h1/2

    new_contour = c1
    
    for c2 in contours:
        if c2 is c1:
           continue

        x2,y2,w2,h2 = cv2.boundingRect(c2)

        cx2 = x2+w2/2
        cy2 = y2+h2/2

        if cv2.contourArea(c2) < 50:
            continue
        
        if abs(cx1-cx2) < 50 and abs(cy1-cy2) < 5 and w1 > h1:
            new_points = np.concatenate((new_contour,c2), axis=0)
            new_contour = cv2.convexHull(new_points)
            contours.remove(c2)
            continue
        if abs(cx1-cx2) < 5 and abs(cy1-cy2) < 50 and w1 < h1:
            new_points = np.concatenate((new_contour,c2), axis=0)
            new_contour = cv2.convexHull(new_points)
            contours.remove(c2)
    
    _,_,nw,nh = cv2.boundingRect(new_contour)
    
    if nw > 35 or nh > 35:
       filtered_contours.append(new_contour)

  print(f"Found {len(contours)} raw contours. Removing Noise and Filtering by shape...")
  print(f"Found {len(filtered_contours)} boards.")

  all_boards = [] # This will store the (x,y) of all circles from detected boards
  # Loop over every contour found
  for contour in filtered_contours:
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

          cv2.circle(debug, (cx-consts.TARGET_OFFSET*consts.SCALE_FACTOR,cy), 5, (255,0,0), -1) 
          cv2.circle(debug, (cx+consts.TARGET_OFFSET*consts.SCALE_FACTOR,cy), 5, (255,0,0), -1)
      else:
          cv2.circle(debug, (cx,cy-consts.TARGET_OFFSET*consts.SCALE_FACTOR), 5, (255,0,0), -1) 
          cv2.circle(debug, (cx,cy+consts.TARGET_OFFSET*consts.SCALE_FACTOR), 5, (255,0,0), -1)
      #Save the target's coordinates

      cv2.circle(debug, (cx,cy), 5, (255,0,0), -1)
      

      board = [
          (cx/consts.SCALE_FACTOR,cy/consts.SCALE_FACTOR),
          [
              target_1,
              target_2
          ]
      ]

      all_boards.append(board)
    
  debug = cv2.resize(debug, (int(consts.MAP_WIDTH/consts.SCALE_FACTOR),int(consts.MAP_HEIGHT/consts.SCALE_FACTOR)))
  cv2.imshow("Clueboards Found", debug)
  
  return all_boards