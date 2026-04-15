import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Load & train model once at startup ──────────────────────────────────────
DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
)

OBESITY_LABELS = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Obesity Type I",
    3: "Obesity Type II",
    4: "Obesity Type III",
    5: "Overweight Level I",
    6: "Overweight Level II",
}

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

print("Downloading dataset and training model…")

data = pd.read_csv(DATA_URL)

continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

# Remember target mapping before encoding
target_map = dict(enumerate(data['NObeyesdad'].astype('category').cat.categories))
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)

accuracy = round(100 * accuracy_score(y_test, model.predict(X_test)), 2)
print(f"Model ready — Test accuracy: {accuracy}%")

FEATURE_COLUMNS = X.columns.tolist()


def preprocess_input(form):
    """Transform raw form values into the same feature space used for training."""
    age    = float(form['age'])
    height = float(form['height'])
    weight = float(form['weight'])
    fcvc   = float(form['fcvc'])
    ncp    = float(form['ncp'])
    ch2o   = float(form['ch2o'])
    faf    = float(form['faf'])
    tue    = float(form['tue'])

    gender   = form['gender']
    fhwo     = form['family_history']
    favc     = form['favc']
    caec     = form['caec']
    smoke    = form['smoke']
    scc      = form['scc']
    calc     = form['calc']
    mtrans   = form['mtrans']

    # Build a single-row raw dataframe matching the original structure
    raw = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': fhwo,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans,
    }])

    # Scale continuous columns
    cont_vals = scaler.transform(raw[continuous_columns])
    cont_df   = pd.DataFrame(cont_vals, columns=scaler.get_feature_names_out(continuous_columns))

    # Encode categorical columns
    cat_vals = encoder.transform(raw[categorical_columns])
    cat_df   = pd.DataFrame(cat_vals, columns=encoder.get_feature_names_out(categorical_columns))

    # Combine – drop original continuous & categorical, add processed ones
    result = pd.concat([
        raw.drop(columns=continuous_columns + categorical_columns).reset_index(drop=True),
        cont_df,
        cat_df,
    ], axis=1)

    # Reindex to match training feature order (fills any missing with 0)
    result = result.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return result


@app.route('/')
def index():
    return render_template('index.html', accuracy=accuracy)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = preprocess_input(request.form)
        pred_code   = int(model.predict(features)[0])
        proba       = model.predict_proba(features)[0]

        raw_label = target_map.get(pred_code, f"Class {pred_code}")
        label = raw_label.replace("_", " ")

        # Build sorted probability list
        class_probs = [
            {"label": target_map.get(i, f"Class {i}").replace("_", " "), "prob": round(float(p) * 100, 1)}
            for i, p in enumerate(proba)
        ]
        class_probs.sort(key=lambda x: x['prob'], reverse=True)

        return jsonify({
            "success":    True,
            "label":      label,
            "color":      OBESITY_COLORS.get(raw_label, "#6366f1"),
            "icon":       OBESITY_ICONS.get(raw_label, "info"),
            "advice":     OBESITY_ADVICE.get(raw_label, ""),
            "confidence": round(float(proba[pred_code]) * 100, 1),
            "class_probs": class_probs,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
