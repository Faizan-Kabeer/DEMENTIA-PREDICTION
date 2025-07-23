import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix
from mealpy.swarm_based.GWO import OriginalGWO  # Corrected GWO import path
from mealpy.utils.space import FloatVar         # Import for defining bounds
import warnings

warnings.filterwarnings('ignore')

# Set a seed for reproducibility
np.random.seed(42)

# Load dataset
data = pd.read_excel('oasis2.xlsx')

# --- Data Preprocessing ---
data_cleaned = data.drop(columns=['Subject ID', 'MRI ID', 'Hand'])
label_encoder = LabelEncoder()
data_cleaned['M/F'] = label_encoder.fit_transform(data_cleaned['M/F'])
data_cleaned['Group'] = label_encoder.fit_transform(data_cleaned['Group'])
data_cleaned = data_cleaned.dropna()
scaler = MinMaxScaler()
numerical_features = [col for col in data_cleaned.columns if col not in ['M/F', 'Group']]
data_cleaned[numerical_features] = scaler.fit_transform(data_cleaned[numerical_features])

X = data_cleaned.drop(columns=['Group'])
y = data_cleaned['Group']

# Define base models
base_models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
}

# Function to create a more advanced MLP model
def create_mlp_model(params):
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

# --- K-Fold Cross-Validation ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"--- FOLD {fold + 1}/5 ---")
    X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

    # Apply SMOTE to the full training data for this fold
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_full, y_train_full)

    # --- Step 1: Hyperparameter Optimization using GWO ---
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train_resampled, y_train_resampled, test_size=0.2, random_state=42, stratify=y_train_resampled
    )

    base_predictions_train_opt = []
    base_predictions_val_opt = []
    for name, model in base_models.items():
        model.fit(X_train_opt, y_train_opt)
        base_predictions_train_opt.append(model.predict_proba(X_train_opt))
        base_predictions_val_opt.append(model.predict_proba(X_val_opt))

    X_meta_train_opt = np.hstack(base_predictions_train_opt)
    X_meta_val_opt = np.hstack(base_predictions_val_opt)

    def objective_function(solution):
        mlp_model = create_mlp_model(solution)
        mlp_model.fit(X_meta_train_opt, y_train_opt)
        return -mlp_model.score(X_meta_val_opt, y_val_opt)

    # Define the problem for mealpy with the correct structure
    problem_dict = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[10, 10, 10, 0.0001], ub=[200, 200, 200, 0.01]),
        "minmax": "min",
    }

    print("Running GWO for hyperparameter optimization...")
    gwo_optimizer = OriginalGWO(epoch=50, pop_size=30)

    best_agent = gwo_optimizer.solve(problem_dict)
    best_params = best_agent.solution
    best_fitness = best_agent.target.fitness

    print(f"GWO found best params: Neurons={tuple(int(p) for p in best_params[:3])}, LR={best_params[3]:.4f}")

    # --- Step 2: Final Model Training on Full Resampled Data ---
    print("Training final model on full resampled data with optimal parameters...")

    base_predictions_train_full = []
    base_predictions_test = []
    for name, model in base_models.items():
        model.fit(X_train_resampled, y_train_resampled)
        base_predictions_train_full.append(model.predict_proba(X_train_resampled))
        base_predictions_test.append(model.predict_proba(X_test))

    X_meta_train_full = np.hstack(base_predictions_train_full)
    X_meta_test = np.hstack(base_predictions_test)

    final_mlp_model = create_mlp_model(best_params)
    final_mlp_model.fit(X_meta_train_full, y_train_resampled)

    # --- Step 3: Evaluation ---
    y_pred = final_mlp_model.predict(X_meta_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    results.append({"accuracy": accuracy, "confusion_matrix": confusion})
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}\n")

# --- Aggregate and Print Final Results ---
mean_accuracy = np.mean([res['accuracy'] for res in results])
std_accuracy = np.std([res['accuracy'] for res in results])
print("\n--- FINAL RESULTS ---")
print(f"Mean Cross-Validated Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}\n")

print("Confusion Matrices for each fold:")
for idx, res in enumerate(results, 1):
    print(f" Fold {idx}:\n{res['confusion_matrix']}\n" + "-" * 20)