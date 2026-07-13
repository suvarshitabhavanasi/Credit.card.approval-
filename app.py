from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = []

    features.append(float(request.form["ID"]))
    features.append(float(request.form["CODE_GENDER"]))
    features.append(float(request.form["FLAG_OWN_CAR"]))
    features.append(float(request.form["FLAG_OWN_REALTY"]))
    features.append(float(request.form["CNT_CHILDREN"]))
    features.append(float(request.form["AMT_INCOME_TOTAL"]))
    features.append(float(request.form["NAME_INCOME_TYPE"]))
    features.append(float(request.form["NAME_EDUCATION_TYPE"]))
    features.append(float(request.form["NAME_FAMILY_STATUS"]))
    features.append(float(request.form["NAME_HOUSING_TYPE"]))
    features.append(float(request.form["DAYS_BIRTH"]))
    features.append(float(request.form["DAYS_EMPLOYED"]))
    features.append(float(request.form["FLAG_MOBIL"]))
    features.append(float(request.form["FLAG_WORK_PHONE"]))
    features.append(float(request.form["FLAG_PHONE"]))
    features.append(float(request.form["FLAG_EMAIL"]))
    features.append(float(request.form["OCCUPATION_TYPE"]))
    features.append(float(request.form["CNT_FAM_MEMBERS"]))

    final_input = np.array(features).reshape(1, -1)

    prediction = model.predict(final_input)

    if prediction[0] == 1:
        result = "Credit Card Rejected"
    else:
        result = "Credit Card Approved"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)