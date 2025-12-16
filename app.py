from flask import Flask, request, jsonify
from flask_cors import CORS # Needed to allow the HTML to talk to Python
import pickle
import numpy as np

app = Flask(__name__)
CORS(app) # Enables the connection

# 1. Load your trained model (Saved from your notebook)
# with open('my_model.pkl', 'rb') as f:
#     model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # 2. Get data from the HTML form
    data = request.get_json()
    
    # Extract variables (ensure these match the IDs in your HTML config)
    area = float(data['area'])
    floors = float(data['floors'])
    bedrooms = float(data['bedrooms'])
    
    # 3. Make Prediction
    # This is where your ML model runs. 
    # For now, using a dummy calculation so you can test it without a model file.
    prediction = (area * 1000) + (floors * 5000) + (bedrooms * 2000) 
    
    # If using real model:
    # features = np.array([[area, floors, bedrooms]])
    # prediction = model.predict(features)[0]

    # 4. Send back to HTML
    return jsonify({'prediction': f"${prediction:,.2f}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)