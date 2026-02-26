import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def run_preprocess():
    #load Data (skip first row if it's just ID descriptions)
    #Note: Adjust 'header=1' if your CSV has a double header, commonly found in this dataset
    df = pd.read_csv("data/raw/credit_card_defaulters.csv", header=1)

    #renaming columns for easier coding
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    
    #drop ID column, it doesn't help prediction
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    #split features (X) and target (y)
    y = df["default"]
    X = df.drop(columns=["default"])

    #split into Train/Test
    #stratify ensures we have the same % of defaults in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #scaling (standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #save everything
    #we save as DataFrames to keep column names, which helps with SHAP later
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    
    #saving the scaler because crucial for the API later.
    joblib.dump(scaler, "models/scaler.pkl")
    
    print("Preprocessing complete. Data and scaler saved.")

if __name__ == "__main__":
    run_preprocess()