import os
from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('.././models/best_model.pkl')

# Define column names
num_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
cat_cols = ['Type']

# Initialize preprocessor with all possible categories for 'Type'
all_types = [['L', 'M', 'H']]
cat_transformer = Pipeline(steps=[('label', OrdinalEncoder(categories=all_types))])
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols), ('cat', cat_transformer, cat_cols)])

# Create a DataFrame with all possible categories to fit the preprocessor
initial_data = pd.DataFrame({
    'Air temperature [K]': [300, 310, 320], 
    'Process temperature [K]': [310, 320, 330],
    'Rotational speed [rpm]': [1500, 1600, 1700], 
    'Torque [Nm]': [40, 50, 60], 
    'Tool wear [min]': [100, 150, 200], 
    'Type': ['L', 'M', 'H']
})

# Fit the preprocessor with initial data containing all possible categories for 'Type'
preprocessor.fit(initial_data)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # If a CSV file is uploaded
        if 'file' in request.files:
            file = request.files['file']
            df = pd.read_csv(file)
            
            # Ensure all required columns are present
            df = df[num_cols + cat_cols]

            # Preprocess the data
            df_processed = preprocessor.transform(df)

            # Make predictions
            predictions = model.predict(df_processed)
            probabilities = model.predict_proba(df_processed)

            # Add predictions and probabilities to the DataFrame
            df['Predicted Failure'] = predictions
            df['Failure Probability'] = probabilities[:, 1]

            # Save the output CSV file
            output_file = 'output.csv'
            df.to_csv(output_file, index=False)

            return send_file(output_file, as_attachment=True)

        else:
            # If individual form data is submitted
            form_data = request.form

            # Prepare input data from form
            input_data = {col: [float(form_data[col])] for col in num_cols}
            input_data['Type'] = [form_data['Type']]
            df = pd.DataFrame(input_data)

            # Preprocess the input data
            df_processed = preprocessor.transform(df)

            # Make prediction
            prediction = model.predict(df_processed)
            print("the prediciton is ",prediction)
            probability = model.predict_proba(df_processed)[0, 1]

            # Generate graph
            # plt.figure(figsize=(6, 4))
            # plt.bar(['No Failure', 'Failure'], [1 - probability, probability], color=['green', 'red'])
            # plt.title('Failure Prediction')
            # plt.ylabel('Probability')
            # plt.savefig('static/prediction_graph.png')

            return render_template('index.html', 
                                   prediction=prediction[0],
                                   probability=probability,
                                   graph='static/prediction_graph.png',
                                   input_data=input_data)

    return render_template('index.html')

if __name__ == '__main__':
    # Ensure static folder exists
    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(debug=True)