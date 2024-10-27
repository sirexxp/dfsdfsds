import argparse
import sys
import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
import threading
import os
import requests
from flask import Flask, Response

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
url = 'http://192.168.0.105/800x600.jpg'

# Flask app to serve the processed image
app = Flask(__name__)

# Global variable to store the latest processed image
latest_frame = None

def run(model: str, max_results: int, score_threshold: float, 
        camera_id: int, width: int, height: int, discord_webhook_url: str) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
      model: Name of the TFLite object detection model.
      max_results: Max number of detection results.
      score_threshold: The score threshold of detection results.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
      discord_webhook_url: The Discord webhook URL to send the ngrok link.
    """

    global latest_frame

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    detection_frame = None
    detection_result_list = []

    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, latest_frame

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        detection_result_list.append(result)
        COUNTER += 1

    # Initialize the object detection model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=save_result
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # Start ngrok tunnel
    from pyngrok import ngrok

    # Make sure ngrok is authenticated
    # You can authenticate ngrok once by running `ngrok authtoken YOUR_AUTHTOKEN` in the terminal
    public_url = ngrok.connect(5000).public_url
    print(f"Ngrok tunnel opened at {public_url}")

    # Send the public URL to Discord webhook
    data = {
        "content": f"Der Stream ist verfügbar unter: {public_url}/video_feed"
    }
    response = requests.post(discord_webhook_url, json=data)
    if response.status_code == 204:
        print("Ngrok-Link erfolgreich an Discord gesendet.")
    else:
        print(f"Fehler beim Senden an Discord: {response.status_code}")

    # Start Flask app in a separate thread
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}).start()

    # Continuously capture images and run inference
    while True:
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        image = cv2.imdecode(imgNp, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run object detection using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image.copy()
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if detection_result_list:
            current_frame = visualize(current_frame, detection_result_list[0])
            detection_frame = current_frame
            detection_result_list.clear()

        if detection_frame is not None:
            # Update the latest frame for the Flask app
            latest_frame = detection_frame.copy()
            # Display the frame locally if needed
            cv2.imshow('object_detection', detection_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cv2.destroyAllWindows()
    ngrok.kill()

# Flask route to serve the video feed
@app.route('/video_feed')
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def get_frame():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='efficientdet_lite0.tflite')
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
        default=0.25)
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
    parser.add_argument(
        '--discordWebhookUrl',
        help='Discord webhook URL to send the ngrok link.',
        required=True)
    args = parser.parse_args()

    run(args.model, int(args.maxResults),
        args.scoreThreshold, int(args.cameraId), args.frameWidth, args.frameHeight, args.discordWebhookUrl)


if __name__ == '__main__':
    main()
