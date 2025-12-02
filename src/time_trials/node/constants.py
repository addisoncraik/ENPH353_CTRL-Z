import numpy as np
import tensorflow as tf

MAP_WIDTH = 2500
MAP_HEIGHT = 1200
SCALE_FACTOR = 2

DRONE_COLOR = np.array([17, 225, 255])
HEAD_COLOR = np.array([255, 255, 0])

TARGET_RADIUS = 20 #measured in scaled map pixels
TARGET_OFFSET = 30 #measured in scaled map pixels
TARGET_HEIGHT = 0.2 #measured in meters

CRUISE_ALTITUDE = 0.5 #measured in meters

CHAR_WIDTH = 100
CHAR_HEIGHT = 145

CLUEBOARD_WIDTH = 500
CLUEBOARD_HEIGHT = 350

MODEL_PATH = '/home/fizzer/competition_ws/src/time_trials/node/read_boards/clue_character_model_v6.tflite' #Need to Upload this file or change to your own

try:
    CNN_MODEL = tf.lite.Interpreter(model_path=MODEL_PATH)
    CNN_MODEL.allocate_tensors()

    CNN_INPUTS = CNN_MODEL.get_input_details()
    CNN_OUTPUTS = CNN_MODEL.get_output_details()

    print(f"CNN Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    # Use a dummy variable if model load fails, allowing script to run without prediction
    CNN_MODEL = None
    print(f"ERROR: Could not load CNN model. Prediction will be disabled.")
    print(e)
    

DEBUG = True