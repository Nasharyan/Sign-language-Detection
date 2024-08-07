"""
Flask application for hand gesture recognition using a pre-trained model and MediaPipe.
"""

import pickle
import cv2
import numpy as np
import mediapipe as mp
import time
import logging
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Load the pre-trained model
try:
    with open('./model.p', 'rb') as model_file:
        model_dict = pickle.load(model_file)
        model = model_dict['model']
except (FileNotFoundError, KeyError) as error:
    logging.error("Error loading the model: %s", error)
    raise error

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Gesture labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '4', 27: '3',
    28: '2', 29: 'Rock'
}

# Initialize video capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    logging.error("Error: Could not open video device.")
    raise RuntimeError("Error: Could not open video device.")



def generate_frames():
    """Generate frames from the video capture device and resize them."""
    frame_width = 1900  # Set the desired width
    frame_height = 1200  # Set the desired height

    while True:
        start_time = time.time()  # Start time for response time measurement
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Failed to capture image.")
            break

        # Resize frame
        frame = cv2.resize(frame, (frame_width, frame_height))

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_coords = []
            y_coords = []

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for landmark in hand_landmarks.landmark:
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_coords))
                    data_aux.append(landmark.y - min(y_coords))

            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                x1 = int(min(x_coords) * width) - 10
                y1 = int(min(y_coords) * height) - 10
                x2 = int(max(x_coords) * width) - 10
                y2 = int(max(y_coords) * height) - 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(
                    frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            except (ValueError, KeyError) as model_error:
                logging.error("Prediction error: %s", model_error)

        end_time = time.time()  # End time for response time measurement
        response_time = end_time - start_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the camera and release resources."""
    global cap
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Camera stopped."})


@app.route('/performance')
def performance():
    """Return performance metrics."""
    # Static values for demonstration
    performance_metrics = {
        "response_time": "50ms",
        "accuracy": "95%",
        "latency": "100ms"
    }
    return jsonify(performance_metrics)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    try:
        app.run(debug=True)
    except Exception as app_error:
        logging.error("Application error: %s", app_error)
    finally:
        cap.release()
        cv2.destroyAllWindows()
