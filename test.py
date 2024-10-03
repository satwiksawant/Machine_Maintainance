import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the saved model
model = joblib.load('./models/best_model.pkl')

# Define column names
num_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
cat_cols = ['Type']

# Initialize preprocessor
all_types = [['L', 'M', 'H']]
cat_transformer = Pipeline(steps=[('label', OrdinalEncoder(categories=all_types))])
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols), ('cat', cat_transformer, cat_cols)])

# Create test data
test_data = pd.DataFrame({
    'Air temperature [K]': [300, 310, 320], 
    'Process temperature [K]': [310, 320, 330],
    'Rotational speed [rpm]': [1500, 1600, 1700], 
    'Torque [Nm]': [40, 50, 60], 
    'Tool wear [min]': [100, 150, 200], 
    'Type': ['L', 'M', 'H']
})

# Fit the preprocessor with the test data
preprocessor.fit(test_data)

# Preprocess the test data
test_data_processed = preprocessor.transform(test_data)

# Make predictions
predictions = model.predict(test_data_processed)
probabilities = model.predict_proba(test_data_processed)

# Print results
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Sample {i+1}:")
    print(f"Input: {test_data.iloc[i].to_dict()}")
    print(f"Prediction: {'Failure' if pred == 1 else 'No Failure'}")
    print(f"Probability of Failure: {prob[1]:.4f}")
    print()

# Test a sample that should likely result in no failure
no_failure_sample = pd.DataFrame({
    'Air temperature [K]': [297.4], 
    'Process temperature [K]': [308.7],
    'Rotational speed [rpm]': [2874], 
    'Torque [Nm]': [4.2], 
    'Tool wear [min]': [118], 
    'Type': ['L']
})

no_failure_processed = preprocessor.transform(no_failure_sample)
no_failure_pred = model.predict(no_failure_processed)
no_failure_prob = model.predict_proba(no_failure_processed)

print("Sample that should likely result in no failure:")
print(f"Input: {no_failure_sample.iloc[0].to_dict()}")
print(f"Prediction: {'Failure' if no_failure_pred[0] == 1 else 'No Failure'}")
print(f"Probability of Failure: {no_failure_prob[0][1]:.4f}")
print(f"All",no_failure_pred)