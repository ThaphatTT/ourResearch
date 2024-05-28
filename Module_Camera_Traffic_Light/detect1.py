import argparse
import sys
import time

import cv2
import mediapipe as mp
import pygame
import threading

from gpiozero import LED
from time import sleep

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize
from picamera2 import Picamera2

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
global detecting
path = "/home/admin/Desktop/ourResearch/Module_Camera_Traffic_Light/"
sound_files = ["StopWaiting.mp3", "PrepareToCross.mp3", "CanCrossTheRoad.mp3"]
global count_detect_object

green = LED(18)
yellow = LED(23)
red = LED(27)

picam2 = Picamera2()
picam2.preview_configuration.main.size = (2304,1296)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

def run(model: str, max_results: int, score_threshold: float, 
        camera_id: int, width: int, height: int) -> None:
  red.on()
  yellow.on()
  green.on()
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
  AlwayOpenGreenLight()


  # Visualization parameters
  row_size = 50  # pixels
  left_margin = 24  # pixels
  text_color = (0, 255, 0)  # black
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
  while True:
    im= picam2.capture_array()  
#    success, image = cap.read()
    image=cv2.resize(im,(640,480)) 
    image = cv2.flip(image, 1)
    
    roi = image[:, image.shape[1]//2:]
    
    cv2.rectangle(image, (image.shape[1]//2, 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
    
    # Convert the image from BGR to RGB as required by the TFLite model.
    if frame_count % detect_every_n_frames == 0:
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
            elif time.time() - start_time >= 4 and detecting and not sound_playing:
                playSoundStopWaiting()
                start_time = None
                print(f"start_time after play sound stop waiting: {start_time}")
                print(f"detecting after play sound stop waiting : {detecting}")
                print(f"status Sound playing after play sound stop waiting: {sound_playing}")
            elif time.time() - start_time >= 6.5 and detecting and not sound_prepare :
                playSoundPrepareToCross()
                start_time = None
                sound_prepare = True
                print(f"start_time after play sound prepare the cross: {start_time}")
                print(f"detecting after play sound prepare the cross : {detecting}")
                print(f"status Sound playing after play sound prepare the cross: {sound_playing}")
            elif time.time() - start_time >= 9 and detecting and not sound_cancross:
                playSoundCanCrossTheRoad()
                start_time = None
                detecting = False
                sound_cancross = True
                print(f"start_time after play sound Can cross the Road: {start_time}")
                print(f"detecting after play sound Can cross the Road : {detecting}")
                print(f"status Sound playing after play sound Can cross the Road: {sound_playing}")
        else:
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
#       default='efficientdet_lite0.tflite')
      default='best.tflite')
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
  
def openGreenLight():
    red.on()
    yellow.on()
    green.off()
    
def openYellowLight():
    green.on()
    red.on()
    yellow.off()
    
def openRedLight():
    yellow.on()
    green.on()
    red.off()
    
def AlwayOpenGreenLight():
    yellow.on()
    green.off()
    red.on()

    

def play_soundStop(file):
    global sound_playing
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.stop() 
    pygame.mixer.music.load(path + file)
    openGreenLight()
    pygame.mixer.music.play()
    time.sleep(4) 
    pygame.mixer.music.play()
    sound_playing = True
    

def play_soundPrepare(file):
    global sound_playing
    global detecting
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.stop() 
    pygame.mixer.music.load(path + file)
    openYellowLight()
    pygame.mixer.music.play()
    time.sleep(3)  
    pygame.mixer.music.play()
    if detecting == False :
        AlwayOpenGreenLight()
        print(f"In Sound Prepare Else : {detecting}")
    time.sleep(3)  
    pygame.mixer.music.play()
    if detecting == False :
        AlwayOpenGreenLight()
        print(f"In Sound Prepare Else : {detecting}")
    sound_playing = True

def play_soundCan(file):
    global sound_playing
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.stop() 
    pygame.mixer.music.load(path + file)
    openRedLight()
    pygame.mixer.music.play()
    time.sleep(4)  
    pygame.mixer.music.play()
    time.sleep(4)  
    pygame.mixer.music.play()
    time.sleep(4)  
    pygame.mixer.music.play()
    time.sleep(4)  
    pygame.mixer.music.play()
    time.sleep(4)  
    AlwayOpenGreenLight()
    sound_playing = True

def playSoundStopWaiting():
    threading.Thread(target=play_soundStop, args=(sound_files[0],)).start()

def playSoundPrepareToCross():
    threading.Thread(target=play_soundPrepare, args=(sound_files[1],)).start()

def playSoundCanCrossTheRoad():
    threading.Thread(target=play_soundCan, args=(sound_files[2],)).start()
    
if __name__ == '__main__':
  main()