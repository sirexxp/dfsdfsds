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

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
url = 'http://192.168.2.146/800x600.jpg'

# Flask app to serve the processed image
app = Flask(__name__)

# Global variable to store the latest processed image
latest_frame = None

def visualize(image, detection_result):
    """Zeichnet Begrenzungsrahmen auf das Bild.

    Args:
        image: Das Eingabebild.
        detection_result: Die Liste der Erkennungsergebnisse.

    Returns:
        Annotiertes Bild.
    """
    annotated_image = image.copy()
    for detection in detection_result.detections:
        # Zeichne den Begrenzungsrahmen
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
        cv2.rectangle(annotated_image, start_point, end_point, (0, 255, 0), 2)

        # Zeichne das Label und die Wahrscheinlichkeit
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (int(bbox.origin_x), int(bbox.origin_y - 10))
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return annotated_image

def run(model: str, max_results: int, score_threshold: float, 
        camera_id: int, width: int, height: int, discord_webhook_url: str) -> None:
    """Führt kontinuierlich Inferenz auf Bildern von der Kamera durch.

    Args:
      model: Name des TFLite-Objekterkennungsmodells.
      max_results: Maximale Anzahl von Erkennungsergebnissen.
      score_threshold: Der Schwellwert für die Erkennungsergebnisse.
      camera_id: Die Kamera-ID für OpenCV.
      width: Die Breite des von der Kamera erfassten Rahmens.
      height: Die Höhe des von der Kamera erfassten Rahmens.
      discord_webhook_url: Die Discord-Webhook-URL zum Senden des ngrok-Links.
    """

    global latest_frame

    # Visualisierungsparameter
    row_size = 50  # Pixel
    left_margin = 24  # Pixel
    text_color = (0, 0, 0)  # Schwarz
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    detection_frame = None
    detection_result_list = []

    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, latest_frame

        # Berechne die FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        detection_result_list.append(result)
        COUNTER += 1

    # Initialisiere das Objekterkennungsmodell
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=save_result
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # Starte ngrok-Tunnel
    from pyngrok import ngrok

    # Stelle sicher, dass ngrok authentifiziert ist
    public_url = ngrok.connect(5000).public_url
    print(f"Ngrok-Tunnel geöffnet unter {public_url}")

    # Sende den öffentlichen URL an Discord-Webhook
    data = {
        "content": f"Der Stream ist verfügbar unter: {public_url}/video_feed"
    }
    response = requests.post(discord_webhook_url, json=data)
    if response.status_code == 204:
        print("Ngrok-Link erfolgreich an Discord gesendet.")
    else:
        print(f"Fehler beim Senden an Discord: {response.status_code}")

    # Starte Flask-App in einem separaten Thread
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}).start()

    # Führe kontinuierlich die Erkennung durch
    while True:
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        image = cv2.imdecode(imgNp, 1)

        # Konvertiere das Bild von BGR zu RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Führe die Objekterkennung durch
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Zeige die FPS an
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
            # Aktualisiere den neuesten Frame für die Flask-App
            latest_frame = detection_frame.copy()
            # Zeige den Frame lokal an (optional)
            cv2.imshow('object_detection', detection_frame)

        # Beende das Programm, wenn die ESC-Taste gedrückt wird
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cv2.destroyAllWindows()
    ngrok.kill()

# Flask-Route zum Bereitstellen des Video-Feeds
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
        help='Pfad zum Objekterkennungsmodell.',
        required=False,
        default='efficientdet_lite0.tflite')
    parser.add_argument(
        '--maxResults',
        help='Maximale Anzahl von Erkennungsergebnissen.',
        required=False,
        default=5)
    parser.add_argument(
        '--scoreThreshold',
        help='Der Schwellwert für die Erkennungsergebnisse.',
        required=False,
        type=float,
        default=0.25)
    parser.add_argument(
        '--cameraId', help='ID der Kamera.', required=False, type=int, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Breite des von der Kamera erfassten Rahmens.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Höhe des von der Kamera erfassten Rahmens.',
        required=False,
        type=int,
        default=480)
    parser.add_argument(
        '--discordWebhookUrl',
        help='Discord-Webhook-URL zum Senden des ngrok-Links.',
        required=True)
    args = parser.parse_args()

    run(args.model, int(args.maxResults),
        args.scoreThreshold, int(args.cameraId), args.frameWidth, args.frameHeight, args.discordWebhookUrl)

if __name__ == '__main__':
    main()
