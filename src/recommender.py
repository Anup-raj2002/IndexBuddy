import os
import joblib
import pandas as pd

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'index_buddy_et.joblib')

# Load the trained Extra Trees model
try:
    et_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    et_model = None

def evaluate_index_viability(parsed_data, total_rows, matching_rows):
    """
    Evaluates if an index should be created using the Extra Trees ML model.
    Returns a dictionary formatted for the Streamlit UI.
    """
    if et_model is None:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run src/model_trainer.py first.")

    # Extract parsed AST data
    table = parsed_data.get('table_name', 'unknown_table')
    column = parsed_data.get('candidate_column', 'unknown_column')
    has_join = parsed_data.get('has_join', 0)
    is_range_scan = parsed_data.get('is_range_scan', 0)

    # 1. Calculate Selectivity
    selectivity = matching_rows / max(1, total_rows)

    # 2. Prepare features exactly as the model was trained
    features = pd.DataFrame([{
        'table_rows': total_rows,
        'selectivity': selectivity,
        'has_join': has_join,
        'is_range_scan': is_range_scan
    }])

    # 3. Model Inference
    prediction = et_model.predict(features)[0]
    
    # Extract prediction confidence probability
    probabilities = et_model.predict_proba(features)[0]
    confidence = probabilities[1] if prediction == 1 else probabilities[0]

    # 4. Generate DDL
    if prediction == 1 and column != 'unknown_column':
        ddl_statement = f"CREATE INDEX idx_{table}_{column} ON {table}({column});"
    else:
        ddl_statement = "-- No index recommended. Write overhead outweighs read benefits."

    # 5. Return payload for app.py
    return {
        'selectivity': selectivity,
        'is_beneficial': bool(prediction == 1),
        'confidence': float(confidence),
        'ddl_statement': ddl_statement
    }