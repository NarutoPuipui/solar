from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model5.pkl'  # Ensure this is the updated model path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data (updated for new input structure)
    # Adjust this based on the features used in your model
    form_values = [float(x) for x in request.form.values()]  # Assuming numerical inputs
    final_features = [np.array(form_values)]  # Convert inputs to NumPy array
    
    # Make prediction
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Predicted Energy Output: {output} kW')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/developers')
def contact():
    return render_template('developers.html')

if __name__ == "__main__":
    app.run(debug=True)
