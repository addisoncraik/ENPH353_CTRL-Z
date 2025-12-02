import math

import constants as consts

def find_target(babyDrone, boards):
    cx, cy, _ = babyDrone

    if boards == []:
        return None, None

    distances = []

    for board in boards:
        bx, by = board[0]

        distances.append(math.sqrt((cx - bx) ** 2 + (cy - by) ** 2))

    closest_board = boards[distances.index(min(distances))]


    distances = []
    targets = closest_board[1]

    for target in targets:
        tx, ty = target

        distance = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)

        distances.append(distance)

    closest_target = targets[distances.index(min(distances))]

    return closest_board[0], closest_target

def is_at_target(babyDrone, target):
    cx, cy, _ = babyDrone
    tx, ty = target

    dist = math.sqrt((cx-tx)**2 + (cy-ty)**2)
    print("corrected distance " + str(dist))

    return dist < consts.TARGET_RADIUS