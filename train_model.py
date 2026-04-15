"""
Run this script once locally to train and save the model artifacts.
Outputs: model.pkl, scaler.pkl, encoder.pkl, metadata.pkl
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
)

print("Downloading dataset…")
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

target_map = dict(enumerate(data['NObeyesdad'].astype('category').cat.categories))
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training model…")
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)

accuracy = round(100 * accuracy_score(y_test, model.predict(X_test)), 2)
print(f"Test accuracy: {accuracy}%")

# Save all artifacts
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump({
    'target_map': target_map,
    'continuous_columns': continuous_columns,
    'categorical_columns': categorical_columns,
    'feature_columns': X.columns.tolist(),
    'accuracy': accuracy,
}, 'metadata.pkl')

print("Saved: model.pkl, scaler.pkl, encoder.pkl, metadata.pkl")
