import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, classification_report

def run_evaluate():
    #loading data and model
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    
    model = joblib.load("models/xgboost.pkl")

    #get predictions
    #we want probabilities (risk score), not just 0/1
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    #calculate metrics
    auc = roc_auc_score(y_test, probs)
    print(f"XGBoost ROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    run_evaluate()