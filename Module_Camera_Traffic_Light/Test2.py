# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run object detection."""

import argparse
import sys
import time
import pygame
import cv2
import mediapipe as mp
import threading

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
global detecting
path = "C:/Users/Dollars/Desktop/ourResearch/Module_Camera_Traffic_Light/"
sound_files = ["StopWaitingEng.mp3", "PrepareToCrossEng.mp3", "CanCrossTheRoadEng.mp3" ]
global count_detect_object
def run(model: str, max_results: int, score_threshold: float, 
        camera_id: int, width: int, height: int) -> None:
  global start_time
  global detecting
  global sound_playing
  global sound_prepare
  global sound_cancross
  sound_playing = False
  start_time = None
  detecting = False
  sound_prepare = False
  sound_cancross = False
  
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    max_results: Max number of detection results.
    score_threshold: The score threshold of detection results.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
  """

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 50  # pixels
  left_margin = 24  # pixels
  text_color = (0, 255, 0)  # green
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  detection_frame = None
  detection_result_list = []

  pygame.init()

  def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
      global FPS, COUNTER, START_TIME
      

      # Calculate the FPS
      if COUNTER % fps_avg_frame_count == 0:
          FPS = fps_avg_frame_count / (time.time() - START_TIME)
          START_TIME = time.time()

      detection_result_list.append(result)
      COUNTER += 1

  # Initialize the object detection model
  base_options = python.BaseOptions(model_asset_path=model)
  options = vision.ObjectDetectorOptions(base_options=base_options,
                                         running_mode=vision.RunningMode.LIVE_STREAM,
                                         max_results=max_results, score_threshold=score_threshold,
                                         result_callback=save_result)
  detector = vision.ObjectDetector.create_from_options(options)


  frame_count = 0
  detect_every_n_frames = 2

    
  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
        sys.exit(
            'ERROR: Unable to read from webcam. Please verify your webcam settings.'
        )

    # Resize the image
    image = cv2.resize(image, (640, 480))

    # Flip the image
    image = cv2.flip(image, 1)

    # Define the region of interest (ROI) - the right half of the image
    roi = image[:, image.shape[1]//2:]

    # Draw a green rectangle around the ROI on the original image
    cv2.rectangle(image, (image.shape[1]//2, 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)

    if frame_count % detect_every_n_frames == 0:
    # Convert the ROI from BGR to RGB as required by the TFLite model.
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_roi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)

    # Run object detection using the model.
        detector.detect_async(mp_roi, time.time_ns() // 1_000_000)

    frame_count += 1
    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(FPS)
    text_location = (left_margin, row_size)
    current_frame = image
    cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                font_size, text_color, font_thickness, cv2.LINE_AA)


    if detection_result_list:
        # Visualize the detection results on the ROI, not the original image
        roi = visualize(roi, detection_result_list[0])
        # Replace the ROI on the original image with the visualized ROI
        if detection_result_list[0].detections: 
            if start_time is None:
                start_time = time.time()
                detecting = True
                print(f"start_time before detect: {start_time}")
                print(f"detecting: {detecting}")
                print(f"status Sound playing before detect: {sound_playing}")
            elif time.time() - start_time >= 2 and detecting and not sound_playing:
                playSoundStopWaiting()
                start_time = None
                print(f"start_time after play sound stop waiting: {start_time}")
                print(f"detecting after play sound stop waiting : {detecting}")
                print(f"status Sound playing after play sound stop waiting: {sound_playing}")
            elif time.time() - start_time >= 4 and detecting and not sound_prepare :
                playSoundPrepareToCross()
                start_time = None
                sound_prepare = True
                print(f"start_time after play sound prepare the cross: {start_time}")
                print(f"detecting after play sound prepare the cross : {detecting}")
                print(f"status Sound playing after play sound prepare the cross: {sound_playing}")
            elif time.time() - start_time >= 8 and detecting and not sound_cancross:
                playSoundCanCrossTheRoad()
                start_time = None
                detecting = False
                sound_cancross = True
                print(f"start_time after play sound prepare the cross: {start_time}")
                print(f"detecting after play sound prepare the cross : {detecting}")
                print(f"status Sound playing after play sound prepare the cross: {sound_playing}")
        else:
            pygame.mixer.music.pause()
            start_time = None
            detecting = False
            sound_playing = False
            sound_prepare = False
            sound_cancross = False
            print(f"start_time: {start_time}")
            print(f"detecting: {detecting}")
        current_frame[:, current_frame.shape[1]//2:] = roi
        detection_frame = current_frame
        detection_result_list.clear()
       


    if detection_frame is not None:
        cv2.imshow('object_detection', detection_frame)
      

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
        break



  detector.close()
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
#      default='efficientdet_lite0.tflite')
      default='C:/Users/Dollars/Desktop/ourResearch/Module_Camera_Traffic_Light/best.tflite')
  parser.add_argument(
      '--maxResults',
      help='Max number of detection results.',
      required=False,
      default=5)
  parser.add_argument(
      '--scoreThreshold',
      help='The score threshold of detection results.',
      required=False,
      type=float,
      default=0.6)
  # Finding the camera ID can be very reliant on platform-dependent methods. 
  # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0. 
  # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
  # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  args = parser.parse_args()

  run(args.model, int(args.maxResults),
      args.scoreThreshold, int(args.cameraId), args.frameWidth, args.frameHeight)
  
        

def play_sound(file):
    global sound_playing
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.stop() 
    pygame.mixer.music.load(path + file)
    pygame.mixer.music.play(-1)

   
    sound_playing = True

def playSoundStopWaiting():
    threading.Thread(target=play_sound, args=(sound_files[0],)).start()

def playSoundPrepareToCross():
    threading.Thread(target=play_sound, args=(sound_files[1],)).start()

def playSoundCanCrossTheRoad():
    threading.Thread(target=play_sound, args=(sound_files[2],)).start()

if __name__ == '__main__':
  main()