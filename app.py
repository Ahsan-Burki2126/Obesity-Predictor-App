import pandas as pd
import joblib
import os
from flask import Flask, render_template, request, jsonify
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

OBESITY_COLORS = {
    "Insufficient_Weight":  "#3b82f6",
    "Normal_Weight":        "#22c55e",
    "Overweight_Level_I":   "#f59e0b",
    "Overweight_Level_II":  "#f97316",
    "Obesity_Type_I":       "#ef4444",
    "Obesity_Type_II":      "#dc2626",
    "Obesity_Type_III":     "#991b1b",
}

OBESITY_ICONS = {
    "Insufficient_Weight":  "monitor_weight",
    "Normal_Weight":        "check_circle",
    "Overweight_Level_I":   "warning",
    "Overweight_Level_II":  "warning",
    "Obesity_Type_I":       "dangerous",
    "Obesity_Type_II":      "dangerous",
    "Obesity_Type_III":     "dangerous",
}

OBESITY_ADVICE = {
    "Insufficient_Weight":  "Your weight is below the healthy range. Consider increasing caloric intake with nutrient-dense foods and consult a dietitian.",
    "Normal_Weight":        "You are at a healthy weight. Keep up the great work with your diet and physical activity!",
    "Overweight_Level_I":   "You are slightly above the healthy weight range. Small changes in diet and increased physical activity can help.",
    "Overweight_Level_II":  "You are moderately above the healthy weight range. Consider consulting a healthcare provider for a personalised plan.",
    "Obesity_Type_I":       "Your BMI indicates Obesity Type I. A structured diet and exercise programme under medical supervision is recommended.",
    "Obesity_Type_II":      "Your BMI indicates Obesity Type II. Please consult a healthcare professional for a comprehensive weight-management plan.",
    "Obesity_Type_III":     "Your BMI indicates Obesity Type III. Immediate medical consultation is strongly advised.",
}

# ── Load pre-trained artifacts ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model    = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
scaler   = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
encoder  = joblib.load(os.path.join(BASE_DIR, 'encoder.pkl'))
metadata = joblib.load(os.path.join(BASE_DIR, 'metadata.pkl'))

target_map         = metadata['target_map']
continuous_columns = metadata['continuous_columns']
categorical_columns = metadata['categorical_columns']
FEATURE_COLUMNS    = metadata['feature_columns']
accuracy           = metadata['accuracy']


def preprocess_input(form):
    """Transform raw form values into the same feature space used for training."""
    raw = pd.DataFrame([{
        'Gender': form['gender'],
        'Age':    float(form['age']),
        'Height': float(form['height']),
        'Weight': float(form['weight']),
        'family_history_with_overweight': form['family_history'],
        'FAVC':   form['favc'],
        'FCVC':   float(form['fcvc']),
        'NCP':    float(form['ncp']),
        'CAEC':   form['caec'],
        'SMOKE':  form['smoke'],
        'CH2O':   float(form['ch2o']),
        'SCC':    form['scc'],
        'FAF':    float(form['faf']),
        'TUE':    float(form['tue']),
        'CALC':   form['calc'],
        'MTRANS': form['mtrans'],
    }])

    cont_vals = scaler.transform(raw[continuous_columns])
    cont_df   = pd.DataFrame(cont_vals, columns=scaler.get_feature_names_out(continuous_columns))

    cat_vals = encoder.transform(raw[categorical_columns])
    cat_df   = pd.DataFrame(cat_vals, columns=encoder.get_feature_names_out(categorical_columns))

    result = pd.concat([
        raw.drop(columns=continuous_columns + categorical_columns).reset_index(drop=True),
        cont_df,
        cat_df,
    ], axis=1)

    return result.reindex(columns=FEATURE_COLUMNS, fill_value=0)


@app.route('/')
def index():
    return render_template('index.html', accuracy=accuracy)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features  = preprocess_input(request.form)
        pred_code = int(model.predict(features)[0])
        proba     = model.predict_proba(features)[0]

        raw_label = target_map.get(pred_code, f"Class {pred_code}")
        label     = raw_label.replace("_", " ")

        class_probs = [
            {"label": target_map.get(i, f"Class {i}").replace("_", " "), "prob": round(float(p) * 100, 1)}
            for i, p in enumerate(proba)
        ]
        class_probs.sort(key=lambda x: x['prob'], reverse=True)

        return jsonify({
            "success":     True,
            "label":       label,
            "color":       OBESITY_COLORS.get(raw_label, "#6366f1"),
            "icon":        OBESITY_ICONS.get(raw_label, "info"),
            "advice":      OBESITY_ADVICE.get(raw_label, ""),
            "confidence":  round(float(proba[pred_code]) * 100, 1),
            "class_probs": class_probs,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
