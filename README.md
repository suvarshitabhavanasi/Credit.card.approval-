# Credit Card Approval Prediction

This project trains a machine learning model to predict whether a credit card application should be approved. The trained pipeline is saved in `models/credit_approval_pipeline.joblib` and is used by the Flask app in `app.py`.

## Project Files

- `train.py` trains the model and saves the pipeline artifact.
- `app.py` starts the Flask web app and loads the saved model.
- `templates/index.html` contains the user interface.
- `static/style.css` contains the page styling.
- `data/application_record.csv` and `data/credit_record.csv` are the training datasets.
- `nb.ipynb` and `notebook.ipynb` are notebooks for analysis and experimentation.

## Requirements

Install Python 3.10 or newer. Then install the project dependencies with:

```powershell
python -m pip install -r requirements.txt
```

If you are using a virtual environment, activate it first, then run the same command.

## Step-by-Step Setup

1. Open a terminal in the project folder.
2. Create a virtual environment if you want to keep dependencies isolated:

```powershell
python -m venv .venv
```

3. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Install the requirements:

```powershell
python -m pip install -r requirements.txt
```

5. Train the model and generate `models/credit_approval_pipeline.joblib`:

```powershell
python train.py
```

6. Start the web app:

```powershell
python app.py
```

7. Open the local address shown in the terminal, usually `http://127.0.0.1:5000`.

## How To Run Each File

### `train.py`

Run this file first if the model file is missing or you want to retrain the pipeline:

```powershell
python train.py
```

This script reads the CSV files in `data/`, trains the model, prints evaluation metrics, and saves the trained pipeline in `models/credit_approval_pipeline.joblib`.

### `app.py`

Run the app after the model has been created:

```powershell
python app.py
```

The app loads the saved model and starts a Flask server in debug mode.

### `nb.ipynb` and `notebook.ipynb`

Open either notebook in Jupyter or VS Code and run the cells manually if you want to inspect the data or experiment with the model. These files do not need to be run for the Flask app to work.

## Recommended Order

1. Install dependencies.
2. Run `train.py`.
3. Run `app.py`.
4. Use the browser form to test predictions.

## Notes

- If `app.py` says the trained model is missing, run `python train.py` first.
- The app depends on the files inside `data/` and the generated model inside `models/`.
- If you change the training code, rerun `train.py` before starting the app again.
