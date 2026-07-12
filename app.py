from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_approval_pipeline.joblib"

FEATURE_COLUMNS = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "FLAG_MOBIL",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
]


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()


@app.route("/")
def home():
    return render_template("index.html", model_ready=model is not None)


def build_input_frame(form):
    age = int(form["AGE"])
    employment_years = int(form["EMPLOYMENT_YEARS"])
    mobile_number = form["MOBILE_NUMBER"].strip()
    email = form["EMAIL"].strip()

    data = {
        "CODE_GENDER": form["CODE_GENDER"],
        "FLAG_OWN_CAR": form["FLAG_OWN_CAR"],
        "FLAG_OWN_REALTY": form["FLAG_OWN_REALTY"],
        "CNT_CHILDREN": int(form["CNT_CHILDREN"]),
        "AMT_INCOME_TOTAL": float(form["AMT_INCOME_TOTAL"]),
        "NAME_INCOME_TYPE": form["NAME_INCOME_TYPE"],
        "NAME_EDUCATION_TYPE": form["NAME_EDUCATION_TYPE"],
        "NAME_FAMILY_STATUS": form["NAME_FAMILY_STATUS"],
        "NAME_HOUSING_TYPE": form["NAME_HOUSING_TYPE"],
        "DAYS_BIRTH": -(age * 365),
        "DAYS_EMPLOYED": -(employment_years * 365),
        "FLAG_MOBIL": 1,
        "FLAG_WORK_PHONE": 0,
        "FLAG_PHONE": int(bool(mobile_number)),
        "FLAG_EMAIL": int(bool(email)),
        "OCCUPATION_TYPE": form["OCCUPATION_TYPE"],
        "CNT_FAM_MEMBERS": float(form["CNT_FAM_MEMBERS"]),
    }

    return pd.DataFrame([data], columns=FEATURE_COLUMNS)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html",
            error="Trained model not found. Run train.py to generate models/credit_approval_pipeline.joblib.",
            model_ready=False,
        )

    try:
        input_frame = build_input_frame(request.form)
        prediction = model.predict(input_frame)[0]
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Prediction failed: {exc}",
            model_ready=True,
        )

    if prediction == 1:
        result = "✅ CREDIT CARD APPROVED"
        message = "Congratulations! The applicant satisfies the approval criteria."
        result_class = "approved"
    else:
        result = "❌ CREDIT CARD REJECTED"
        message = "The applicant does not satisfy the approval criteria."
        result_class = "rejected"

    return render_template(
        "index.html",
        prediction=result,
        message=message,
        result_class=result_class,
        model_ready=True,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
