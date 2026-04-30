from flask import Flask, request, jsonify

app = Flask(__name__)

# 🔥 conversión de clase YOLO a nivel de riesgo
def riesgo_ph(ph_class):
    if ph_class == "8":
        return 3  # infección fuerte
    elif ph_class == "7_8":
        return 2  # probable infección
    elif ph_class == "6_7":
        return 1  # leve
    elif ph_class == "5_6":
        return 0  # sano
    else:
        return 0

@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.json

    temp = float(data["temperatura"])
    ph_class = data["ph"]   # viene como clase YOLO: "8", "7_8", etc.

    riesgo = riesgo_ph(ph_class)

    score = 0

    # 🌡️ lógica de temperatura
    if temp > 38:
        score += 2
    elif temp > 37.5:
        score += 1

    # 🧪 lógica de pH (YOLO)
    score += riesgo

    # 🧠 DECISIÓN FINAL
    if score >= 3:
        resultado = "Infección probable 🚨"
    elif score == 2:
        resultado = "Riesgo medio ⚠️"
    else:
        resultado = "Normal 👍"

    return jsonify({
        "resultado": resultado,
        "score": score,
        "ph_clase": ph_class,
        "temperatura": temp
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
