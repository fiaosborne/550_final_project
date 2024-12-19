import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from supervised.automl import AutoML
import argparse
from scipy.sparse import csr_matrix

def main(behavioural_csv_path):
    '''
    Runs AutoML jar library on feature matrix located at behavioural_csv_path

    Example usage: python ./beavioural_gender_automljar.py ./your_data_csv
    '''
    # Load the behavioral data
    df = pd.read_csv(behavioural_csv_path, sep='\t', index_col=0)
    print('file read')
    # Separate the first column as target (y)
    for column in df.columns[1:]:  # Skip the first column (target)
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
    target_array = df.iloc[:, 0].values  # This gets the first column as a numpy array (labels)

    # Get the features (exclude the first column)
    feature_matrix = csr_matrix(df.iloc[:, 1:].values)  # This gets all columns except the first one as a numpy array (features)

    # Convert to sparse format (if desired)
    #sparse_matrix = csr_matrix(feature_matrix)
    print('spare matrix made')
    # Select features (X) and target (y)
    #X = behavioural_data[behavioural_data.columns[1:]]  
    #y = behavioural_data['label']  # Target column

    # Initialize and run AutoML
    automl = AutoML(mode="Perform", train_ensemble=False, explain_level=2)
    print('model initiated')
    automl.fit(feature_matrix, target_array)

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