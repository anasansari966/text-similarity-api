from flask import Flask, request, jsonify
import joblib
from sentence_transformers import util
import os

model = joblib.load("similarity_model.joblib")

app = Flask(__name__)

@app.route("/", methods=["POST"])
def get_similarity():
    try:
        data = request.get_json()
        text1 = data.get("text1")
        text2 = data.get("text2")

        embeddings = model.encode([text1, text2])
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return jsonify({"similarity score": round(score, 4)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)