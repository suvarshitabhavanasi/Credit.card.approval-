from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

APPLICATION_COLUMNS = [
    "ID",
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

CATEGORICAL_COLUMNS = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

NUMERIC_COLUMNS = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "FLAG_MOBIL",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "CNT_FAM_MEMBERS",
]


def build_target(credit_frame: pd.DataFrame) -> pd.DataFrame:
    approved = credit_frame.copy()
    approved["Approved"] = approved["STATUS"].apply(lambda value: 0 if value in {"2", "3", "4", "5"} else 1)
    approved = approved.groupby("ID", as_index=False)["Approved"].min()
    return approved


def load_training_frame() -> pd.DataFrame:
    data_dir = PROCESSED_DATA_DIR if (PROCESSED_DATA_DIR / "application_record.csv").exists() else DATA_DIR
    application = pd.read_csv(data_dir / "application_record.csv")
    credit = pd.read_csv(data_dir / "credit_record.csv")
    target = build_target(credit)

    frame = application.merge(target, on="ID", how="inner")
    frame = frame[APPLICATION_COLUMNS + ["Approved"]].copy()
    frame["OCCUPATION_TYPE"] = frame["OCCUPATION_TYPE"].fillna("Unknown")
    frame = frame.drop_duplicates(subset=["ID"]).reset_index(drop=True)
    return frame


def build_pipeline() -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    frame = load_training_frame()
    features = frame.drop(columns=["Approved", "ID"])
    target = frame["Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(classification_report(y_test, predictions, digits=4))

    joblib.dump(pipeline, MODEL_DIR / "credit_approval_pipeline.joblib")
    print(f"Saved model to {MODEL_DIR / 'credit_approval_pipeline.joblib'}")


if __name__ == "__main__":
    main()
