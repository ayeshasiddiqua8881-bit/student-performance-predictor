"""
Student Performance Predictor — Flask Web Application
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = [
    "study_hours",
    "attendance",
    "prev_score",
    "assignments_done",
    "sleep_hours",
    "extra_curricular",
]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate & parse
        values = []
        for feat in FEATURES:
            val = data.get(feat)
            if val is None:
                return jsonify({"error": f"Missing field: {feat}"}), 400
            values.append(float(val))

        X = np.array(values).reshape(1, -1)

        prediction   = int(model.predict(X)[0])          # 0 = Fail, 1 = Pass
        proba        = model.predict_proba(X)[0]          # [P(Fail), P(Pass)]
        confidence   = round(float(proba[prediction]) * 100, 1)
        pass_prob    = round(float(proba[1]) * 100, 1)
        fail_prob    = round(float(proba[0]) * 100, 1)

        label = "Pass" if prediction == 1 else "Fail"

        # Simple tip based on weakest input
        tips = generate_tips(dict(zip(FEATURES, values)))

        return jsonify({
            "prediction":  label,
            "confidence":  confidence,
            "pass_prob":   pass_prob,
            "fail_prob":   fail_prob,
            "tips":        tips,
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


def generate_tips(data: dict) -> list[str]:
    tips = []
    if data["study_hours"] < 3:
        tips.append("📚 Study at least 3–5 hours daily for consistent improvement.")
    if data["attendance"] < 75:
        tips.append("🏫 Try to maintain 75%+ attendance — it directly impacts results.")
    if data["prev_score"] < 50:
        tips.append("📝 Revisit core subjects to build a stronger foundation.")
    if data["assignments_done"] < 60:
        tips.append("✅ Complete more assignments — they reinforce your learning.")
    if data["sleep_hours"] < 6:
        tips.append("😴 Get 7–8 hours of sleep; it improves memory and focus.")
    if not tips:
        tips.append("🌟 Great profile! Keep maintaining your study habits.")
    return tips


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
