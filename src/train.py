import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def run_train():
    #load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    #train Logistic Regression
    print("Training Logistic Regression...")
    log_model = LogisticRegression(random_state=42, max_iter=1000)
    log_model.fit(X_train, y_train)
    joblib.dump(log_model, "models/logistic.pkl")

    #train XGBoost
    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, "models/xgboost.pkl")

    print("Models trained and saved in /models")

if __name__ == "__main__":
    run_train()