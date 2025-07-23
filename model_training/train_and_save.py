import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import warnings
import os

# --- Configuration & Setup ---
warnings.filterwarnings('ignore')
np.random.seed(42) # Ensure reproducibility

# --- Model Hyperparameters ---
# **IMPORTANT**: Set these to the best parameters found during your cross-validation.
# Using the parameters from Fold 2 of your results as an example.
BEST_MLP_PARAMS = [56, 81, 82, 0.0054] # [hidden_layer_1, hidden_layer_2, hidden_layer_3, learning_rate]

# --- Function to Create MLP Model ---
# This function must be identical to the one in your training script.
def create_mlp_model(params):
    """Instantiates an MLPClassifier with a given set of hyperparameters."""
    return MLPClassifier(
        hidden_layer_sizes=(int(params[0]), int(params[1]), int(params[2])),
        learning_rate_init=params[3],
        max_iter=500,
        random_state=42,
        solver='adam',
        activation='relu',
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.2
    )

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Final Model Training and Saving Process ---")

    # --- 1. Load and Preprocess Data ---
    try:
        data = pd.read_excel("../oasis2.xlsx")
        print("Dataset 'oasis2.xlsx' loaded successfully.")
    except FileNotFoundError:
        print("Error: 'oasis2.xlsx' not found. Please ensure the file is in the correct directory.")
        exit()

    # Create a directory to store the deployment artifacts
    output_dir = "../web_app/deployment_artifacts"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Models and preprocessors will be saved in the '{output_dir}/' directory.")

    # Drop irrelevant columns
    data_cleaned = data.drop(columns=['Subject ID', 'MRI ID', 'Hand'])

    # Handle missing values by dropping rows
    data_cleaned = data_cleaned.dropna()
    
    # --- 2. Prepare and Save Encoders ---
    # We save the encoders fitted on the original data to correctly process new inputs.
    
    # Encoder for 'M/F' column
    mf_encoder = LabelEncoder()
    data_cleaned['M/F'] = mf_encoder.fit_transform(data_cleaned['M/F'])
    joblib.dump(mf_encoder, os.path.join(output_dir, 'encoder_mf.joblib'))
    print("Saved 'M/F' LabelEncoder as 'encoder_mf.joblib'.")

    # Encoder for the target 'Group' column (to convert predictions back to labels)
    group_encoder = LabelEncoder()
    data_cleaned['Group'] = group_encoder.fit_transform(data_cleaned['Group'])
    joblib.dump(group_encoder, os.path.join(output_dir, 'encoder_group.joblib'))
    print("Saved 'Group' LabelEncoder as 'encoder_group.joblib'.")

    # --- 3. Prepare and Save Scaler ---
    numerical_features = ['Visit', 'MR Delay', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
    scaler = MinMaxScaler()
    data_cleaned[numerical_features] = scaler.fit_transform(data_cleaned[numerical_features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    print("Saved MinMaxScaler as 'scaler.joblib'.")

    # --- 4. Prepare Data for Training ---
    X = data_cleaned.drop(columns=['Group'])
    y = data_cleaned['Group']
    
    # Ensure column order is consistent for deployment
    feature_order = list(X.columns)
    joblib.dump(feature_order, os.path.join(output_dir, 'feature_order.joblib'))
    print(f"Saved feature order: {feature_order}")

    # Apply SMOTE to the entire dataset to handle class imbalance for final training
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Applied SMOTE. Original dataset size: {X.shape}, Resampled dataset size: {X_resampled.shape}")

    # --- 5. Train and Save Base Models ---
    base_models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "SVC": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }

    base_model_predictions = []
    for name, model in base_models.items():
        print(f"Training final {name} model on all data...")
        model.fit(X_resampled, y_resampled)
        
        # Save the trained model
        model_filename = f"model_{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, os.path.join(output_dir, model_filename))
        print(f"Saved {name} model as '{model_filename}'.")
        
        # Generate predictions to be used as features for the meta-model
        base_model_predictions.append(model.predict_proba(X_resampled))

    # --- 6. Train and Save the Final MLP Meta-Model ---
    # Create the meta-features by stacking the base model predictions
    X_meta = np.hstack(base_model_predictions)

    print("Training final MLP meta-model with the best hyperparameters...")
    final_mlp_model = create_mlp_model(BEST_MLP_PARAMS)
    final_mlp_model.fit(X_meta, y_resampled)

    # Save the final meta-model
    joblib.dump(final_mlp_model, os.path.join(output_dir, 'model_mlp_meta.joblib'))
    print("Saved MLP meta-model as 'model_mlp_meta.joblib'.")

    print("\n--- Process Complete ---")
    print("All model components have been trained on the full dataset and saved successfully.")