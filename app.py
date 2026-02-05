from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect 30 input features in correct order
    features = []
    for i in range(1, 31):
        features.append(float(request.form[f"f{i}"]))

    # Convert to numpy array
    input_data = np.array([features])

    # Apply same scaling used during training
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)

    # Interpret result (check your dataset mapping)
    if prediction[0] == 1:
        result = "Cancer Detected"
    else:
        result = "No Cancer"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)

