from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
basedir = os.path.abspath(os.path.dirname(__file__))
model = joblib.load(os.path.join(basedir, 'model_decision_tree_air.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    hasil = ''
    data = request.form
    
    ph = data.get('ph')
    suhu = data.get('suhu')
    tds = data.get('tds')

    # Contoh input: {"features": [6.8, 2.8, 4.8, 1.4]}
    features = np.array([[ph, suhu, tds]]).reshape(1, -1)
    prediction = model.predict(features)
    
    if prediction == 0:
        hasil = 'Baik'
    elif prediction == 1:
        hasil = 'Buruk'
    elif prediction == 2:
        hasil = 'Optimal'
    else:
        hasil = 'Tidak Diketahui'
    

    return jsonify({'prediction': hasil})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
