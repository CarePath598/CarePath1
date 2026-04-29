from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.json
    
    temp = data["temperatura"]
    ph = data["ph"]

    # 🔹 MODELO SIMPLE (puedes mejorar después)
    if temp > 37.5 or ph > 7.5:
        resultado = "Infección probable"
    else:
        resultado = "Normal"

    return jsonify({"resultado": resultado})

app.run(host="0.0.0.0", port=5000)