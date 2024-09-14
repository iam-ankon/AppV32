from flask import Flask, render_template, request, jsonify
from scipy.signal import find_peaks
import tensorflow as tf
import numpy as np
import requests
from gtts import gTTS

# Load model and weights
model = tf.keras.models.load_model('my_model.keras')
model.load_weights('model_weights.keras')

# Initialize global variables
tmp = []
movementList = []
alphabets = ['downdog', 'goddess', 'plank', 'tree', 'warrior']
landmarks_data = []
output = ''
startPrediction = True

app = Flask(__name__, static_url_path='/static')

# Function to correct text using LanguageTool API
def correct_text(text):
    api_url = "https://languagetool.org/api/v2/check"
    params = {"text": text, "language": "en-US"}

    try:
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            data = response.json()
            corrected_text = text
            offset = 0

            for match in data['matches']:
                message = match['message']
                offset_start = match['offset']
                offset_end = offset_start + match['length']
                replacement = match['replacements'][0]['value']

                corrected_text = (corrected_text[:offset_start + offset] +
                                  replacement +
                                  corrected_text[offset_end + offset:])
                offset += len(replacement) - match['length']

            return corrected_text
        else:
            print(f"Error: Unable to connect to the LanguageTool API. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Function to generate output
def outputGeneration():
    global output
    global landmarks_data

    # Process landmarks if there's enough data
    if len(landmarks_data) > 0:
        # Prepare the data for prediction
        resFrame = np.array(landmarks_data)
        print(f"Shape of resFrame for prediction: {resFrame.shape}")  # Debug

        # Ensure resFrame is valid
        if resFrame.size == 0:
            print("Error: resFrame is empty.")
            return

        # Try making a prediction with the model
        try:
            res = model.predict(resFrame)
            print(f"Model prediction results: {res}")  # Debug

            # Get the most probable pose for the last frame
            predicted_pose = alphabets[np.argmax(res[-1])]
            print(f"Predicted posture: {predicted_pose}")  # Debug
            output = predicted_pose
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            output = "Error"
    else:
        output = "No landmarks data"

    # Clear landmarks_data after processing
    landmarks_data = []


@app.route("/")
def home():
    return render_template('ankon.html')

@app.route("/send_landmarks", methods=["POST"])
def receive_landmarks():
    global landmarks_data
    data = request.get_json()
    landmarks = data.get("landmarks")
    if landmarks:
        keypoints = []
        for mark in landmarks:
            keypoints.extend([mark["x"], mark["y"], mark["z"]])
        landmarks_data.append(keypoints)

        # After receiving landmarks, generate output immediately
        outputGeneration()

        return jsonify({"message": "Landmarks data received successfully"})
    else:
        return jsonify({"message": "No landmarks data received"}), 400

@app.route("/get_landmarks", methods=["GET"])
def get_landmarks():
    global output
    # Return the detected posture or a default waiting message
    return jsonify(string=output if output else "Waiting for posture...")

# Run the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
