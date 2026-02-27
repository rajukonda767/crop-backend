from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

from predict import predict_disease
from nlp_engine import generate_telugu_advice
from voice import speak_telugu

app = Flask(__name__)
CORS(app)

# ===============================
# SAFE PATH SETUP (VERY IMPORTANT)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ===============================
# HEALTH CHECK
# ===============================
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running"}), 200


# ===============================
# AUDIO FILE SERVING
# ===============================
@app.route("/response.mp3")
def serve_audio():
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "response.mp3")

    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype="audio/mpeg")

    return jsonify({"error": "Audio file not found"}), 404


# ===============================
# PREDICTION ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # ===== AI Prediction =====
        disease, confidence = predict_disease(filepath)

        # ===== NLP Advice =====
        message, severity, treatments, preventions = generate_telugu_advice(
            disease,
            confidence
        )

        # ===== Voice Generation =====
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "response.mp3")

        try:
            speak_telugu(message, audio_path)
        except Exception as e:
            print("Audio failed:", e)

        return jsonify({
            "disease": disease,
            "confidence": round(confidence * 100, 2),
            "severity": severity,
            "telugu_message": message,
            "treatments": treatments,
            "preventions": preventions,
            "voice_file": request.host_url + "response.mp3"
        })

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({"error": str(e)}), 500


# ===============================
# ERROR HANDLERS
# ===============================
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal Server Error"}), 500


# ===============================
# LOCAL RUN
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)