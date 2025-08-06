from flask import Flask, render_template, request, jsonify
import numpy as np
from joblib import load
import os
from werkzeug.utils import secure_filename
from plant_disease_classifier import PlantDiseaseClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the Random Forest model for crop prediction
try:
    model_path = os.path.join(os.path.dirname(__file__), 'crop_prediction_model.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found")
    RF = load(model_path)
    if not hasattr(RF, 'predict'):
        raise AttributeError("Invalid model format: model lacks predict method")
except Exception as e:
    print(f"Error loading crop prediction model: {str(e)}")
    RF = None

# Initialize disease classifier
try:
    model_path = os.path.join(os.path.dirname(__file__), 'plant_disease_prediction_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found")
    classifier = PlantDiseaseClassifier()
    classifier.load_model(model_path, as_ensemble=True)  # Load as ensemble model
    classifier.models.append(classifier.main_model)  # Add main model to ensemble list
    print(f"Successfully loaded disease classifier from {model_path}")
except Exception as e:
    print(f"Error loading disease classifier: {str(e)}")
    classifier = None

def validate_input(value, name, min_val=0, max_val=None):
    try:
        val = float(value)
        if val < min_val or (max_val is not None and val > max_val):
            raise ValueError(
                f"{name} must be between {min_val} and {max_val if max_val is not None else 'infinity'}")
        return val
    except ValueError:
        raise ValueError(f"{name} must be a valid number")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop_prediction')
def crop_prediction():
    return render_template('crop_prediction.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    if RF is None:
        return jsonify({'error': 'Crop prediction model not loaded. Please ensure the model file exists and is valid.'})

    try:
        # Validate and get all input parameters
        features = [
            validate_input(request.form['nitrogen'], 'Nitrogen', min_val=0, max_val=240),
            validate_input(request.form['phosphorus'], 'Phosphorus', min_val=0, max_val=245),
            validate_input(request.form['potassium'], 'Potassium', min_val=0, max_val=205),
            validate_input(request.form['ph'], 'pH', min_val=0, max_val=14),
            validate_input(request.form['ec'], 'EC', min_val=0),
            validate_input(request.form['sulfur'], 'Sulfur', min_val=0),
            validate_input(request.form['copper'], 'Copper', min_val=0),
            validate_input(request.form['iron'], 'Iron', min_val=0),
            validate_input(request.form['manganese'], 'Manganese', min_val=0),
            validate_input(request.form['zinc'], 'Zinc', min_val=0),
            validate_input(request.form['boron'], 'Boron', min_val=0)
        ]

        # Make prediction
        prediction = RF.predict([features])
        return jsonify({
            'success': True,
            'prediction': prediction[0]
        })

    except ValueError as ve:
        return jsonify({
            'success': False,
            'error': str(ve)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}'
        })

@app.route('/disease_detection')
def disease_detection():
    return render_template('disease_detection.html')

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if classifier is None:
        return jsonify({'error': 'Disease classifier not loaded. Please ensure the model file exists and is valid.'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            predicted_class, confidence = classifier.predict(filepath)
            confidence_value = float(confidence) * 100
            
            return jsonify({
                'success': True,
                'filename': filename,
                'predicted_class': predicted_class,
                'confidence': confidence_value
            })
            
        except Exception as e:
            return jsonify({
                'error': 'An error occurred while analyzing the image',
                'details': str(e)
            }), 500

if __name__ == '__main__':
    app.run(debug=True)