import pandas as pd
import joblib
from fairlearn.metrics import demographic_parity_difference
from scipy.stats import ks_2samp

def run_monitoring():
    print("Running Model Monitoring...\n")
    
    #load Data and Model
    #load the raw test data here so we have the original 'sex' and 'age' columns
    df = pd.read_csv("data/raw/credit_card_defaulters.csv", header=1)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    X = df.drop(columns=["default payment next month"])
    y = df["default payment next month"]
    
    model = joblib.load("models/xgboost.pkl")
    scaler = joblib.load("models/scaler.pkl")
    
    #scale data for predictions
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    
    #fairness check
    # SEX: 1 = Male, 2 = Female
    sensitive_feature = X['SEX']
    dp_diff = demographic_parity_difference(y, preds, sensitive_features=sensitive_feature)
    
    print("--- FAIRNESS REPORT ---")
    print(f"Demographic Parity Difference (Gender): {dp_diff:.4f}")
    if dp_diff < 0.05:
        print("Model is fair across genders (Difference < 5%)")
    else:
        print("Warning: Model shows demographic bias.")

    #data drift check
    #simulate "new" incoming data vs "old" training data using age
    old_data = X['AGE'][:15000] # First half
    new_data = X['AGE'][15000:] # Second half (simulating new applicants)
    
    #Kolmogorov-Smirnov test for distribution shift
    statistic, p_value = ks_2samp(old_data, new_data)
    
    print("\n--- DATA DRIFT REPORT (Feature: AGE) ---")
    print(f"KS Statistic: {statistic:.4f}, P-value: {p_value:.4f}")
    if p_value > 0.05:
        print("No significant data drift detected.")
    else:
        print("Warning: Data distribution has shifted!")

if __name__ == "__main__":
    run_monitoring()