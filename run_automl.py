import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from supervised.automl import AutoML
import argparse

def main(behavioural_csv_path):
    '''
    Runs AutoML jar library on feature matrix located at behavioural_csv_path

    Example usage: python ./beavioural_gender_automljar.py ./your_data_csv
    '''
    # Load the behavioral data
    behavioural_data = pd.read_csv(behavioural_csv_path, index_col=0, sep='\t')

    # Convert boolean columns to integer (0 or 1)
    bool_cols = behavioural_data.select_dtypes(include=['bool']).columns
    behavioural_data[bool_cols] = behavioural_data[bool_cols].astype(int)

    # Select features (X) and target (y)
    X = behavioural_data[behavioural_data.columns[1:]]  
    y = behavioural_data['label']  # Target column

    # Initialize and run AutoML
    automl = AutoML(mode="Perform", train_ensemble=False, explain_level=2)
    automl.fit(X, y)

    # Predict probabilities and classes
    y_pred_prob = automl.predict_proba(X)
    y_pred = automl.predict(X)

    # Calculate AUROC and print the result
    print(f"AUROC: {roc_auc_score(y, y_pred):.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoML on behavioral data.")
    parser.add_argument("behavioural_csv_path", help="Path to the behavioral variables file (TSV format).")

    args = parser.parse_args()

    main(args.behavioural_csv_path)
#endregion