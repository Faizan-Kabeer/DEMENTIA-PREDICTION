from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration ---
# Define the directory where the saved model artifacts are located.
ARTIFACTS_DIR = "deployment_artifacts"

# --- Load Model Components ---
def load_model_components(directory):
    """Loads all necessary model files from the specified directory."""
    components = {}
    try:
        # Base models
        components['rf_model'] = joblib.load(os.path.join(directory, 'model_random_forest.joblib'))
        components['svc_model'] = joblib.load(os.path.join(directory, 'model_svc.joblib'))
        components['xgb_model'] = joblib.load(os.path.join(directory, 'model_xgboost.joblib'))
        
        # Meta-model
        components['mlp_model'] = joblib.load(os.path.join(directory, 'model_mlp_meta.joblib'))
        
        # Preprocessors and feature order
        components['scaler'] = joblib.load(os.path.join(directory, 'scaler.joblib'))
        components['mf_encoder'] = joblib.load(os.path.join(directory, 'encoder_mf.joblib'))
        components['group_encoder'] = joblib.load(os.path.join(directory, 'encoder_group.joblib'))
        components['feature_order'] = joblib.load(os.path.join(directory, 'feature_order.joblib'))
        
        print("All model components loaded successfully.")
        return components
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not load model components. {e}")
        print(f"Please ensure the '{directory}/' folder exists and contains all required .joblib files.")
        return None

# Load all components when the application starts
model_components = load_model_components(ARTIFACTS_DIR)

# --- API and Frontend Routes ---

@app.route('/', methods=['GET'])
def home():
    """
    This is the main route. It renders the index.html page, which serves as the user interface.
    """
    # Flask will look for 'index.html' in a folder named 'templates'
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    This is the API endpoint that receives data from the frontend, runs the prediction,
    and returns the result.
    """
    if model_components is None:
        return jsonify({"error": "Model components are not loaded. The server is not ready for predictions."}), 503

    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "Invalid input: No JSON data provided."}), 400

    try:
        input_df = pd.DataFrame([json_data])

        # --- Preprocessing Pipeline ---
        try:
            input_df['M/F'] = model_components['mf_encoder'].transform(input_df['M/F'])
        except ValueError as e:
            return jsonify({"error": f"Invalid value for 'M/F'. Expected one of {model_components['mf_encoder'].classes_}. Details: {e}"}), 400

        numerical_features = model_components['scaler'].feature_names_in_
        input_df[numerical_features] = model_components['scaler'].transform(input_df[numerical_features])
        
        input_df = input_df[model_components['feature_order']]

        # --- Prediction Pipeline ---
        rf_pred_proba = model_components['rf_model'].predict_proba(input_df)
        svc_pred_proba = model_components['svc_model'].predict_proba(input_df)
        xgb_pred_proba = model_components['xgb_model'].predict_proba(input_df)

        meta_features = np.hstack([rf_pred_proba, svc_pred_proba, xgb_pred_proba])

        final_prediction_encoded = model_components['mlp_model'].predict(meta_features)
        
        final_prediction_label = model_components['group_encoder'].inverse_transform(final_prediction_encoded)

        # --- Return the result ---
        return jsonify({
            "prediction_label": final_prediction_label[0],
            "prediction_code": int(final_prediction_encoded[0])
        })

    except KeyError as e:
        return jsonify({"error": f"Missing required feature in input data: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during prediction: {str(e)}"}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
