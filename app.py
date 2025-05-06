from flask import Flask, request, render_template
import pickle
import logging
from sklearn.exceptions import InconsistentVersionWarning
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# Load ML model safely
try:
    with open('best_loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Mapping dictionaries for human-readable inputs
INPUT_MAPPINGS = {
    'Gender': {'male': 1, 'female': 0, 'm': 1, 'f': 0},
    'Married': {'yes': 1, 'no': 0, 'y': 1, 'n': 0},
    'Education': {'graduate': 1, 'not graduate': 0, 'g': 1, 'ng': 0},
    'Self_Employed': {'yes': 1, 'no': 0, 'y': 1, 'n': 0},
    'Property_Area': {'urban': 2, 'rural': 0, 'semiurban': 1}
}

# Expected feature schema with validation rules
FEATURE_SCHEMA = {
    'Gender': {'type': 'categorical', 'options': ['male', 'female', 'm', 'f']},
    'Married': {'type': 'categorical', 'options': ['yes', 'no', 'y', 'n']},
    'Dependents': {'type': 'numeric', 'min': 0, 'max': 5},
    'Education': {'type': 'categorical', 'options': ['graduate', 'not graduate', 'g', 'ng']},
    'Self_Employed': {'type': 'categorical', 'options': ['yes', 'no', 'y', 'n']},
    'ApplicantIncome': {'type': 'numeric', 'min': 0, 'max': 100000},
    'CoapplicantIncome': {'type': 'numeric', 'min': 0, 'max': 50000},
    'LoanAmount': {'type': 'numeric', 'min': 0, 'max': 50000},
    'Loan_Amount_Term': {'type': 'numeric', 'min': 0, 'max': 480},
    'Credit_History': {'type': 'numeric','min': 0, 'max': 5000},
    'Property_Area': {'type': 'categorical', 'options': ['urban', 'rural', 'semiurban']}
}

def preprocess_input(data):
    """Convert human-readable inputs to model-expected numerical values"""
    processed = {}
    errors = []
    
    for field, value in data.items():
        if field not in FEATURE_SCHEMA:
            continue
            
        schema = FEATURE_SCHEMA[field]
        value = str(value).strip().lower()
        
        try:
            if schema['type'] == 'categorical':
                if value not in schema['options']:
                    errors.append(f"Invalid {field}: {value}")
                    continue
                    
                # Special mapping for fields that need conversion
                if field in INPUT_MAPPINGS:
                    processed[field] = INPUT_MAPPINGS[field][value]
                else:
                    processed[field] = value
                    
            elif schema['type'] == 'numeric':
                num_value = int(float(value))
                if num_value < schema['min'] or num_value > schema['max']:
                    errors.append(f"{field} must be between {schema['min']}-{schema['max']}")
                    continue
                processed[field] = num_value
                
            elif schema['type'] == 'binary':
                if value not in ['0', '1']:
                    errors.append(f"{field} must be 0 or 1")
                    continue
                processed[field] = int(value)
            elif field == 'Credit_History':
                num_value = int(float(value))
                if num_value > 500:
                    processed[field] = 1
                elif num_value == 0 or num_value == 1:
                    processed[field] = num_value
                else:
                    errors.append(f"{field} must be between 0 and 1 or above 500 for automatic assignment")
                
        except (ValueError, TypeError):
            errors.append(f"Invalid {field}: {value}")
    
    return processed, errors

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', 
                            prediction="System error: Model not available",
                            error=True)

    try:
        processed_data, errors = preprocess_input(request.form)
        
        if errors:
            return render_template('index.html',
                                prediction="Please correct the following errors: " + ", ".join(errors),
                                error=True)

        # Ensure all required fields are present in correct order
        feature_order = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]
        
        features = [processed_data[field] for field in feature_order]
        prediction = model.predict([features])[0]
        result = "Approved" if prediction == 1 else "Rejected"
        
        return render_template('index.html', 
                            prediction=result,
                            success=True)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return render_template('index.html',
                            prediction="Error processing your request",
                            error=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)