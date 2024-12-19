import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay

def main(behavioural_csv_path, output_directory, learning_rate, depth, rsm):
    # Load behavioral data with Dask (for large datasets)
    behavioural_data = dd.read_csv(behavioural_csv_path, sep='\t')
    print(f"Columns in dataset: {len(behavioural_data.columns)}")
    
    # Convert to Pandas DataFrame (this may be done after sampling or processing)
    behavioural_data = behavioural_data.compute()

    # Select features (X) and target (y)
    X = behavioural_data[behavioural_data.columns[1:]]  # Assuming label is in the first column
    y = behavioural_data['label']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Function to perform cross-validation
    def cross_validate(model, X_input, Y_output):
        kf = KFold(n_splits=5, random_state=1, shuffle=True)
        accuracy_scores = []
        auc_scores = []

        for train_index, val_index in kf.split(X_input):
            X_train_fold, X_val_fold = X_input.iloc[train_index], X_input.iloc[val_index]
            y_train_fold, y_val_fold = Y_output.iloc[train_index], Y_output.iloc[val_index]

            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]

            # Calculate accuracy and AUROC for this fold
            accuracy_scores.append(accuracy_score(y_val_fold, y_pred))
            auc_scores.append(roc_auc_score(y_val_fold, y_pred_proba))

        avg_accuracy = np.mean(accuracy_scores)
        avg_auc = np.mean(auc_scores)
        return avg_accuracy, avg_auc

    # Initialize the model
    model = CatBoostClassifier(learning_rate=learning_rate, depth=depth, rsm=rsm, loss_function='Logloss', verbose=0)

    # Perform cross-validation on the training set
    avg_train_accuracy, avg_train_auc = cross_validate(model, X_train, y_train)
    print(f"Cross-Validation (Training) - Accuracy: {avg_train_accuracy:.4f}, AUROC: {avg_train_auc:.4f}")

    # Train the final model on the entire training set
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    print(f"Test Set - Accuracy: {test_accuracy:.4f}, AUROC: {test_auc:.4f}")

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate and save feature importance
    feature_importance = model.get_feature_importance(Pool(X_train, label=y_train), type="PredictionValuesChange")
    feature_names = X_train.columns

    # Display and save feature importance plot (only top 20 most important features)
    top_n = 20  # Limit to top 20 most important features for visualization
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).head(top_n)
    print(f"Top {top_n} Features by Importance:\n", feature_importance_df)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Features by Importance")
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_directory, "feature_importance.png"))
    plt.close()

    # Calculate and generate SHAP values
    shap_values = model.get_feature_importance(Pool(X_train, label=y_train), type="ShapValues")
    shap_values = shap_values[:, :-1]  # Exclude the last column (base value)

    # SHAP Summary Plot for the top N important features
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.savefig(os.path.join(output_directory, "shap_summary_bar.png"))
    plt.close()

    # SHAP Scatter Plot for the top N important features (limit number of features to plot)
    shap.summary_plot(shap_values, X_train, show=False, max_display=top_n, cmap=plt.get_cmap('viridis'))
    plt.savefig(os.path.join(output_directory, "shap_summary_scatter.png"))
    plt.close()

    # Generate and save partial dependence plots for the top N features
    for feature in feature_importance_df['Feature']:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Generate partial dependence plot for the current feature
        PartialDependenceDisplay.from_estimator(model, X_train, [feature], ax=ax)
        
        # Save each plot as a PNG file
        output_png_path = os.path.join(output_directory, f'pdp_{feature}.png')
        plt.savefig(output_png_path)
        plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run cross-validation and evaluate the final model.")
    parser.add_argument("behavioural_csv_path", help="Path to the behavioral variables file (CSV format).")
    parser.add_argument("output_directory", help="Directory to save the output plots.")
    parser.add_argument("learning_rate", type=float, help="Model Learning Rate")
    parser.add_argument("depth", type=int, help="Tree Depth")
    parser.add_argument("rsm", type=float, help="rsm")
    
    args = parser.parse_args()
    main(args.behavioural_csv_path, args.output_directory, args.learning_rate, args.depth, args.rsm)