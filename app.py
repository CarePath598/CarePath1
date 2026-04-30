from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# 🧠 cargar modelo YOLO
model = YOLO("best.pt")

def riesgo_ph(ph_class):
    if ph_class == "8":
        return 3
    elif ph_class == "7_8":
        return 2
    elif ph_class == "6_7":
        return 1
    elif ph_class == "5_6":
        return 0
    else:
        return 0

@app.route("/predict", methods=["POST"])
def predict():

    # 📸 recibir imagen
    file = request.files["image"]

    # convertir a OpenCV
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 🧠 YOLO inference
    results = model(frame)

    # clase detectada
    cls = str(int(results[0].boxes.cls[0]))

    # 🔥 si también quieres meter temperatura desde frontend
    temp = float(request.form.get("temperatura", 36.5))

    riesgo = riesgo_ph(cls)

    score = 0

    if temp > 38:
        score += 2
    elif temp > 37.5:
        score += 1

    score += riesgo

    if score >= 3:
        resultado = "Infección probable 🚨"
    elif score == 2:
        resultado = "Riesgo medio ⚠️"
    else:
        resultado = "Normal 👍"

    return jsonify({
        "resultado": resultado,
        "score": score,
        "ph_clase": cls,
        "temperatura": temp
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
